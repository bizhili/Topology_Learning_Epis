import torch
import random
import modules.utils as utils


def alpha(i: int, R: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
    """Computes the alpha function for the SIR model.

    Args:
        i: The time step.
        R: The basic reproductive number.
        tau: The mean infectious period.

    Returns:
        The alpha value for the SIR model.
    """
    return 1 - torch.exp(-(R / tau) * i)


def act(
    state: torch.Tensor, R0: torch.Tensor, tau: torch.Tensor, Amat: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Computes the change in the SIR state variables.

    Args:
        state: The current SIR state vector.
        R0: The basic reproductive number.
        tau: The mean infectious period.
        Amat: The contact matrix.

    Returns:
        The new SIR state vector and the change in the number of susceptible individuals.
    """
    deltaSIR = torch.zeros_like(state)
    deltaSIR[0] = -state[0] * torch.matmul(Amat, alpha(state[1], R0, tau))
    deltaSIR[2] = state[1] / tau
    deltaSIR[1] = -deltaSIR[0] - deltaSIR[2]
    return state + deltaSIR, -deltaSIR[0]


def one_strain(
    R0: torch.Tensor,
    tau0: torch.Tensor,
    timeHorizon: int,
    n: int,
    Amat: torch.Tensor,
    time: int = 0,
    fromS: int = 0,
    device: str = "cpu",
) -> torch.Tensor:
    """Simulates the SIR model for a single strain.

    Args:
        R0: The basic reproductive number.
        tau0: The mean infectious period.
        timeHorizon: The number of time steps to simulate.
        n: The population size.
        Amat: The contact matrix.
        time: The time step to start the simulation at.
        fromS: The index of the individual to start the infection from.
        device: The device to run the simulation on.

    Returns:
        A tensor of the change in the number of susceptible individuals for each time step.
    """

    deltaSs = [torch.zeros(n, dtype=torch.float32, device=device)]
    stateNowS = torch.ones(n, dtype=torch.float32, device=device)
    stateNowI = torch.zeros(n, dtype=torch.float32, device=device)
    stateNowR = torch.zeros(n, dtype=torch.float32, device=device)
    stateNow = torch.stack([stateNowS, stateNowI, stateNowR])
    # noise = torch.randn((timeHorizon + 1), dtype=torch.float32, device=device) / 400
    for i in range(timeHorizon):
        if i == time:
            stateNow[0, fromS] = 0.99
            stateNow[1, fromS] = 0.01
            deltaSs[-1][fromS] = 0.01
        stateNow, deltaS = act(stateNow, R0, tau0, Amat)
        deltaSs.append(deltaS.clone())
    deltaSs = torch.stack(deltaSs)  # + noise[:, None]
    return deltaSs.T


def multi_strains(
    G, paras: object, Amat: torch.tensor, intense: int =0, device: str = "cpu") -> torch.Tensor:
    """Simulates the SIR model for multiple strains.

    Args:
        G: ntwworkX object
        paras: A parameter object containing the number of strains, the population size, and the minimum time horizon.
        Amat: The contact matrix.
        intense: low, middle or high degree nodes
        device: The device to run the simulation on.

    Returns:
        A tensor of the change in the number of susceptible individuals for each strain and time step.
    """
    miniTime= 20
    extraDays= 15#15
    timeHorizon= (paras.strains)*miniTime+extraDays
    R0s= paras.R0s
    taus= paras.taus
    randomList= utils.select_nodes_accroding_to_degree(G, paras.strains, intense)
    for i in range(paras.n):
        Amat[i, i]= 1
    deltaSsList= []
    for i in range(paras.strains):
        deltaSsList.append(one_strain(R0s[i], taus[i], timeHorizon, paras.n, Amat, time= i*20, fromS= randomList[i], device= device))
    deltaSsTensor= torch.stack(deltaSsList[0:paras.strains], dim= -1)
    return deltaSsTensor
