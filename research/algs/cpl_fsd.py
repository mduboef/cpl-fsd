import itertools
from typing import Any, Dict

import torch

from research.utils import utils

from .off_policy_algorithm import OffPolicyAlgorithm


# compute the AOT-style FSD loss between two pools of segment advantage scores
def fsdLossFromPools(prefAdvs, rejAdvs, beta=1.0, lossType="logistic"):
	"""
	Relaxed first-order stochastic dominance loss via optimal transport (AOT).

	Enforces that the distribution of preferred segment scores stochastically
	dominates the distribution of rejected segment scores at every quantile.
	Achieved by sorting both pools and pairing equal ranks, then penalising
	any quantile where the preferred score falls below the rejected score.

	prefAdvs : (N,) tensor - advantage scores for preferred segments
	rejAdvs  : (M,) tensor - advantage scores for rejected segments
	beta     : temperature / margin parameter for the chosen loss surrogate
	lossType : one of 'logistic', 'hinge_sq', or 'least_squares'

	Returns (loss, accuracy) where accuracy is the fraction of quantile pairs
	where FSD already holds (preferred quantile > rejected quantile).
	"""
	n = min(len(prefAdvs), len(rejAdvs))

	uSorted, _ = torch.sort(prefAdvs)  # ascending order statistics
	vSorted, _ = torch.sort(rejAdvs)

	# align both to length n via uniform quantile sub-sampling when sizes differ
	if len(uSorted) > n:
		idx = torch.linspace(0, len(uSorted) - 1, n, device=prefAdvs.device).long()
		uSorted = uSorted[idx]
	if len(vSorted) > n:
		idx = torch.linspace(0, len(vSorted) - 1, n, device=rejAdvs.device).long()
		vSorted = vSorted[idx]

	# diff[i] = preferred_quantile[i] - rejected_quantile[i]
	# FSD requires diff[i] >= 0 for all i
	diff = uSorted - vSorted

	if lossType == "logistic":
		# smooth surrogate - continuously penalises every violation
		loss = torch.log1p(torch.exp(-beta * diff)).mean()
	elif lossType == "hinge_sq":
		# squared hinge - dead zone above beta, quadratic penalty below
		loss = torch.clamp(beta - diff, min=0).pow(2).mean()
	elif lossType == "least_squares":
		# IPO-style - penalises both violations and over-satisfaction relative to beta
		loss = (beta - diff).pow(2).mean()
	else:
		raise ValueError(f"Unknown lossType '{lossType}'. Choose from: logistic, hinge_sq, least_squares")

	with torch.no_grad():
		accuracy = (diff > 0).float().mean()

	return loss, accuracy


class CPL_FSD(OffPolicyAlgorithm):
	def __init__(
		self,
		*args,
		alpha: float = 1.0,
		fsdBeta: float = 1.0,
		fsdLossType: str = "logistic",
		bcCoeff: float = 0.0,
		bcData: str = "all",
		bcSteps: int = 0,
		**kwargs,
	) -> None:
		super().__init__(*args, **kwargs)
		assert "encoder" in self.network.CONTAINERS
		assert "actor" in self.network.CONTAINERS
		assert fsdBeta > 0.0, "fsdBeta must be positive"
		self.alpha = alpha
		self.fsdBeta = fsdBeta
		self.fsdLossType = fsdLossType
		self.bcData = bcData
		self.bcSteps = bcSteps
		self.bcCoeff = bcCoeff

	def setup_optimizers(self) -> None:
		params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
		groups = utils.create_optim_groups(params, self.optim_kwargs)
		self.optim["actor"] = self.optim_class(groups)

	def setup_schedulers(self, doNothing=True):
		if doNothing:
			# hold LR constant during BC warm-up steps
			for k in self.schedulers_class.keys():
				if k in self.optim:
					self.schedulers[k] = torch.optim.lr_scheduler.LambdaLR(
						self.optim[k], lr_lambda=lambda x: 1.0
					)
		else:
			self.schedulers = {}
			super().setup_schedulers()

	def _getLogProbs(self, batch):
		# unpack observations and actions depending on batch format
		if "label" in batch:
			# comparison mode: two segments per sample stacked along batch dim
			obs = torch.cat((batch["obs_1"], batch["obs_2"]), dim=0)
			action = torch.cat((batch["action_1"], batch["action_2"]), dim=0)
		else:
			# score mode: single segment per sample
			assert "score" in batch, "batch must contain 'label' or 'score'"
			obs = batch["obs"]
			action = batch["action"]

		obsEnc = self.network.encoder(obs)
		dist = self.network.actor(obsEnc)

		if isinstance(dist, torch.distributions.Distribution):
			lp = dist.log_prob(action)
		else:
			assert dist.shape == action.shape
			# for a deterministic actor, log prob reduces to negative MSE
			lp = -torch.square(dist - action).sum(dim=-1)

		return lp

	def _getBcLoss(self, lp, batch):
		# optionally restrict BC supervision to only the preferred half of the data
		if self.bcData == "pos" and "label" in batch:
			lp1, lp2 = torch.chunk(lp, 2, dim=0)
			lpPos = torch.cat(
				(lp1[batch["label"] <= 0.5], lp2[batch["label"] >= 0.5]), dim=0
			)
			return (-lpPos).mean()
		return (-lp).mean()

	def _splitIntoPools(self, segmentAdv, batch):
		"""
		Split a flat tensor of segment advantage scores into preferred and
		rejected pools.

		Score mode  : segments with score >= median go to preferred pool.
		Comparison mode : unpack (obs_1, obs_2) pairs using the label to
		                  assign each individual segment to the correct pool.
		"""
		if "score" in batch:
			scores = batch["score"].float()
			medianScore = scores.median()
			prefMask = scores >= medianScore
			rejMask = ~prefMask
			prefAdvs = segmentAdv[prefMask]
			rejAdvs = segmentAdv[rejMask]
		else:
			# comparison mode - segmentAdv is (2B,): first B are obs_1, last B are obs_2
			adv1, adv2 = torch.chunk(segmentAdv, 2, dim=0)
			label = batch["label"].float()  # 1.0 = obs_2 is preferred

			# gather preferred segments: obs_2 where label=1, obs_1 where label=0
			prefAdvs = torch.cat([adv2[label >= 0.5], adv1[label <= 0.5]], dim=0)
			# gather rejected segments: the complementary halves
			rejAdvs = torch.cat([adv1[label >= 0.5], adv2[label <= 0.5]], dim=0)

		return prefAdvs, rejAdvs

	def _getFsdLoss(self, batch):
		lp = self._getLogProbs(batch)
		bcLoss = self._getBcLoss(lp, batch)

		# compute per-timestep advantages then sum over the segment dimension
		adv = self.alpha * lp
		segmentAdv = adv.sum(dim=-1)

		prefAdvs, rejAdvs = self._splitIntoPools(segmentAdv, batch)

		# guard: if either pool is empty the FSD loss is undefined - return zero
		if len(prefAdvs) == 0 or len(rejAdvs) == 0:
			device = segmentAdv.device
			zeroLoss = torch.tensor(0.0, device=device, requires_grad=True)
			zeroAcc = torch.tensor(0.0, device=device)
			return zeroLoss, bcLoss, zeroAcc

		fsdLoss, accuracy = fsdLossFromPools(
			prefAdvs, rejAdvs, beta=self.fsdBeta, lossType=self.fsdLossType
		)
		return fsdLoss, bcLoss, accuracy

	def train_step(self, batch: Dict, step: int, total_steps: int) -> Dict:
		fsdLoss, bcLoss, accuracy = self._getFsdLoss(batch)

		# warm-up phase: train on BC only, then switch to FSD objective
		if step < self.bcSteps:
			loss = bcLoss
			fsdLoss = torch.tensor(0.0)
			accuracy = torch.tensor(0.0)
		else:
			loss = fsdLoss + self.bcCoeff * bcLoss

		self.optim["actor"].zero_grad()
		loss.backward()
		self.optim["actor"].step()

		if step == self.bcSteps - 1:
			# reset optimizer and start LR schedule when BC warm-up ends
			del self.optim["actor"]
			params = itertools.chain(self.network.actor.parameters(), self.network.encoder.parameters())
			groups = utils.create_optim_groups(params, self.optim_kwargs)
			self.optim["actor"] = self.optim_class(groups)
			self.setup_schedulers(doNothing=False)

		return dict(fsd_loss=fsdLoss.item(), bc_loss=bcLoss.item(), accuracy=accuracy.item())

	def validation_step(self, batch: Any) -> Dict:
		with torch.no_grad():
			fsdLoss, bcLoss, accuracy = self._getFsdLoss(batch)
		return dict(fsd_loss=fsdLoss.item(), bc_loss=bcLoss.item(), accuracy=accuracy.item())

	def _get_train_action(self, obs: Any, step: int, total_steps: int):
		batch = dict(obs=obs)
		with torch.no_grad():
			action = self.predict(batch, is_batched=False, sample=True)
		return action
