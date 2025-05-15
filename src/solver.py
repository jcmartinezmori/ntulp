import gurobipy as gp
import numpy as np
import pickle
import scipy.sparse as ss
import time
from src.config import *


def main(instance, modelname, **kwargs):

    ts = time.time()

    N, J, K, A, B, V = instance

    m = gp.Model()
    m.Params.OutputFlag = kwargs.get('OutputFlag', 1)
    # m.Params.CrossoverBasis = kwargs.get('CrossoverBasis', 1)
    # m.Params.FeasibilityTol = kwargs.get('FeasibilityTol', 1E-5)
    m.Params.Method = kwargs.get('Method', 1)
    m.Params.NumericFocus = kwargs.get('NumericFocus', 3)
    m.Params.Presolve = kwargs.get('Presolve', 0)
    # m.Params.OptimalityTol = kwargs.get('OptimalityTol', 1E-9)
    m.ModelSense = -1

    m._x = m.addVars(J, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='x')
    m._u = m.addVars(N, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='u')
    for k in K:
        m.addConstr(gp.quicksum(A[k][j] * m._x[j] for j in J) == sum(B[i][k] for i in N))
    for i in N:
        m.addConstr(m._u[i] == gp.quicksum(V[i][j] * m._x[j] for j in J))

    objective = kwargs.get('objective', 'utilitarian')
    if objective == 'utilitarian':
        m.setObjective((gp.quicksum(m._u[i] for i in N))/len(N))
    elif objective == 'maximin':
        z = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY)
        for i in N:
            s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY)
            m.addConstr(m._u[i] - s == z)
        m.setObjective(z)
    else:
        raise Exception('objective {0} not supported'.format(objective))

    cutCount = 0
    blocking_IterCount = -1
    eps, S, = -1, None

    m.optimize()
    x_N = {j: m._x[j].X for j in J}
    u_N = {i: m._u[i].X for i in N}
    kappa = m.getAttr('KappaExact')

    tf = time.time()
    tt = tf - ts
    out = x_N, u_N, tt, cutCount, eps, S, kappa
    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'wb') as file:
        pickle.dump(out, file)

    blocking_TimeLimit = kwargs.get('blocking_TimeLimit', 60)
    blocking_IterLimit = kwargs.get('blocking_IterLimit', 10)
    blocking_EpsLimit = kwargs.get('blocking_EpsLimit', 0)
    blocking_Starts = {tuple(sorted(N))}

    if kwargs.get('IndRat', True):
        for i in N:
            cutCount += 1
            s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY)
            m.addConstr(m._u[i] - s == max(V[i][j]/A[0][j] for j in J))

    m.reset()
    m.optimize()
    x_N = {j: m._x[j].X for j in J}
    u_N = {i: m._u[i].X for i in N}
    kappa = m.getAttr('KappaExact')

    blocking_IterCount += 1
    eps, S = get_blocking(instance, u_N, DepthTimeLimit=blocking_TimeLimit, blocking_Starts=blocking_Starts)
    S = tuple(sorted(S))

    tf = time.time()
    tt = tf - ts
    out = x_N, u_N, tt, cutCount, eps, S, kappa
    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'wb') as file:
        pickle.dump(out, file)

    while eps > blocking_EpsLimit and blocking_IterCount <= blocking_IterLimit:

        print('iterCount: {0}'.format(blocking_IterCount))

        print('... adding cut for current S.')
        intersections = get_intersections(instance, m, u_N, S, LamRatTh=0)
        if intersections is not None:
            cutCount += 1
            s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY)
            m.addConstr(gp.quicksum(m.getVarByName(varname) / lam for varname, lam in intersections) - s == 1)
            min_lam = min(lam for _, lam in intersections)
            max_lam = max(lam for _, lam in intersections)
            print('...... added cut for curr. S with coeff. ratio {0}.'.format(min_lam / max_lam))
        print('... adding cuts for previous S.')
        for ct, prev_S in enumerate(blocking_Starts):
            if prev_S == S:
                continue
            intersections = get_intersections(instance, m, u_N, prev_S, LamRatTh=1E-6)
            if intersections is not None:
                cutCount += 1
                s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY)
                m.addConstr(gp.quicksum(m.getVarByName(varname)/lam for varname, lam in intersections) - s == 1)
                min_lam = min(lam for _, lam in intersections)
                max_lam = max(lam for _, lam in intersections)
                print('...... added cut for prev. S no. {0}. with coeff. ratio {1}'.format(ct, min_lam/max_lam))
        blocking_Starts.add(S)
        print('... solving model.')

        m.reset()
        m.optimize()
        x_N = {j: m._x[j].X for j in J}
        u_N = {i: m._u[i].X for i in N}
        kappa = m.getAttr('KappaExact')

        blocking_IterCount += 1
        eps, S = get_blocking(instance, u_N, DepthTimeLimit=blocking_TimeLimit, blocking_Starts=blocking_Starts)
        S = tuple(sorted(S))

        tf = time.time()
        tt = tf - ts
        out = x_N, u_N, tt, cutCount, eps, S, kappa
        with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'wb') as file:
            pickle.dump(out, file)

    return out


def get_blocking(instance, u_N, **kwargs):

    N, J, K, A, B, V = instance
    Starts = kwargs.get('blocking_Starts', {tuple(sorted(N))})
    Starts = Starts.copy()

    cts = {i: 0 for i in N}
    for Start in Starts:
        for i in Start:
            cts[i] += 1
    weights = {i: np.exp2(-cts[i]) for i in N}
    max_weight = max(weights.values())
    for i in N:
        weights[i] /= max_weight

    for j in J:
        eps_j = 0
        S_j = None
        A_j = A[0][j]
        N_j_V_j = sorted([(i, V[i][j]) for i in N], key=lambda x: x[1], reverse=True)
        for ct in range(1, len(N) + 1):
            if N_j_V_j[ct - 1][1] <= 0:
                break
            eps = min(V_ij * ct / A_j - u_N[i] for i, V_ij in N_j_V_j[:ct])
            if eps > eps_j:
                eps_j = eps
                S_j = tuple(sorted(i for i, _ in N_j_V_j[:ct]))
        if S_j is not None:
            Starts.add(S_j)

    m_S = gp.Model()
    m_S.Params.OutputFlag = kwargs.get('OutputFlag', 1)
    m_S.Params.FeasibilityTol = kwargs.get('FeasibilityTol', 1E-9)
    m_S.Params.MIPFocus = kwargs.get('MIPFocus', 1)
    m_S.Params.NumericFocus = kwargs.get('NumericFocus', 3)
    m_S.NumStart = len(Starts)
    m_S.ModelSense = -1

    m_S._zet = m_S.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='zet')
    m_S._eps = m_S.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='eps')
    m_S._y = m_S.addVars(N, vtype=gp.GRB.BINARY, name='y')
    m_S._x = m_S.addVars(J, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='x')
    m_S._u = m_S.addVars(N, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='u')

    m_S.addConstr(gp.quicksum(m_S._y[i] for i in N) >= 1)
    for k in K:
        m_S.addConstr(gp.quicksum(A[k][j] * m_S._x[j] for j in J) == gp.quicksum(B[i][k] * m_S._y[i] for i in N))
    for i in N:
        m_S.addConstr(m_S._u[i] == gp.quicksum(V[i][j] * m_S._x[j] for j in J))
    for i in N:
        m_S.addGenConstrIndicator(m_S._y[i], True, m_S._zet - weights[i] * m_S._u[i], gp.GRB.LESS_EQUAL, - weights[i] * u_N[i])
        m_S.addGenConstrIndicator(m_S._y[i], True, m_S._eps - m_S._u[i], gp.GRB.LESS_EQUAL, - u_N[i])

    for StartNumber, Start in enumerate(Starts):
        m_S.Params.StartNumber = StartNumber
        m_S.Params.StartNumber = StartNumber
        for i in N:
            if i in Start:
                m_S._y[i].Start = 1
            else:
                m_S._y[i].Start = 0

    m_S.Params.TimeLimit = kwargs.get('DiversityTimeLimit', 300)
    m_S.setObjective(m_S._zet)
    m_S.optimize()
    m_S.addConstr(m_S._zet >= (1-1E-3) * m_S._zet.X)

    m_S.Params.TimeLimit = kwargs.get('DepthTimeLimit', 300)
    m_S.setObjective(m_S._eps)
    m_S.optimize()
    # m_S.addConstr(m_S._eps >= (1-1E-3) * m_S._eps.X)

    # m_S.Params.TimeLimit = 0.2 * kwargs.get('TimeLimit', 60)
    # m_S.setObjective(gp.quicksum(-m_S._y[i] for i in N))
    # m_S.optimize()

    eps = m_S._eps.X
    S = {i for i in N if m_S._y[i].X > 1/2}
    # eps = min(m_S._u[i].X - u_N[i] for i in S)

    return eps, S


def get_intersections(instance, m, u_N, S, **kwargs):

    N, J, K, A, B, V = instance

    m_S = gp.Model()
    m_S.Params.OutputFlag = kwargs.get('OutputFlag', 0)
    # m_S.Params.CrossoverBasis = kwargs.get('CrossoverBasis', 1)
    # m_S.Params.FeasibilityTol = kwargs.get('FeasibilityTol', 1E-9)
    m_S.Params.Method = kwargs.get('Method', 1)
    m_S.Params.NumericFocus = kwargs.get('NumericFocus', 3)
    m_S.Params.Presolve = kwargs.get('Presolve', 0)
    # m_S.Params.OptimalityTol = kwargs.get('OptimalityTol', 1E-9)
    m_S.ModelSense = -1

    m_S._lam = m_S.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='lam')
    m_S._x = m_S.addVars(J, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='x')
    m_S._u = m_S.addVars(S, vtype=gp.GRB.CONTINUOUS, ub=gp.GRB.INFINITY, name='u')

    for k in K:
        m_S.addConstr(gp.quicksum(A[k][j] * m_S._x[j] for j in J) == sum(B[i][k] for i in S))
    for i in S:
        m_S.addConstr(m_S._u[i] == gp.quicksum(V[i][j] * m_S._x[j] for j in J))

    m_S.setObjective(m_S._lam)

    constrs = []
    for i in S:
        constr = m_S.addConstr(m_S._lam <= m_S._u[i] - u_N[i])
        constrs.append(constr)
    m_S.optimize()
    try:
        assert m_S._lam.X > 0
        m_S.remove(constrs)
        m_S.reset()
    except (AttributeError, AssertionError):
        return None

    constr_names_to_indices = {constr.ConstrName: i for i, constr in enumerate(m.getConstrs())}
    basis_mat, basis_varnames = get_basis(m, constr_names_to_indices)

    min_lam, max_lam = 1, 1

    intersections = []
    for var in m.getVars():

        if var.VBasis == BASIC:
            continue

        row_indices, values = [], []
        col = m.getCol(var)
        for i in range(col.size()):
            coeff, constrname = col.getCoeff(i), col.getConstr(i).ConstrName
            row_indices.append(constr_names_to_indices[constrname])
            values.append(coeff)
        col = ss.csr_matrix((values, (row_indices, np.zeros_like(row_indices))), shape=(m.NumConstrs, 1))
        inv_basis_mat_col = ss.linalg.spsolve(basis_mat, col)

        r = {
            basis_varname: -coeff for coeff, basis_varname in zip(inv_basis_mat_col, basis_varnames)
            if basis_varname[0] == 'u'
        }
        # r = {v.VarName: 0 for v in m.getVars()}
        # r[var.VarName] = 1
        # for i, basis_varname in enumerate(basis_varnames):
        #     r[basis_varname] = -inv_basis_mat_col[i]

        if all(r['u[{0}]'.format(i)] <= 0 for i in S if 'u[{0}]'.format(i) in r):
            continue

        constrs = []
        for i in S:
            if 'u[{0}]'.format(i) in r:
                constr = m_S.addConstr(u_N[i] + m_S._lam * r['u[{0}]'.format(i)] <= m_S._u[i])
                constrs.append(constr)
        m_S.optimize()
        if m_S.Status == 2:
            intersections.append((var.VarName, m_S._lam.X))
            if m_S._lam.X < min_lam:
                min_lam = m_S._lam.X
            if m_S._lam.X > m_S._lam.X:
                max_lam = m_S._lam.X
            if min_lam/max_lam < kwargs.get('LamRatTh', 1E-9):
                return None
        m_S.remove(constrs)
        m_S.reset()

    return intersections


def get_basis(m, constr_names_to_indices):

    if m.Status != 2:
        raise Exception('m.Status: {0}'.format(m.Status))

    basis_varnames = []
    col_index = 0
    row_indices, col_indices, values = [], [], []

    for var in m.getVars():
        if var.VBasis == BASIC:
            basis_varnames.append(var.VarName)
            col = m.getCol(var)
            for i in range(col.size()):
                coeff, constrname = col.getCoeff(i), col.getConstr(i).ConstrName
                row_indices.append(constr_names_to_indices[constrname])
                col_indices.append(col_index)
                values.append(coeff)
            col_index += 1

    basis_mat = ss.csr_matrix((values, (row_indices, col_indices)), shape=(m.NumConstrs, m.NumConstrs))

    return basis_mat, basis_varnames

