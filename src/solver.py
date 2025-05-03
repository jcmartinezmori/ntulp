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
    m.Params.FeasibilityTol = kwargs.get('FeasibilityTol', 1E-6)
    m.Params.NumericFocus = kwargs.get('NumericFocus', 3)
    m.Params.OptimalityTol = kwargs.get('OptimalityTol', 1E-9)
    m.ModelSense = -1

    m._x = m.addVars(J, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='x')
    m._u = m.addVars(N, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='u')
    for k in K:
        m.addConstr(gp.quicksum(A[k][j] * m._x[j] for j in J) == sum(B[i][k] for i in N))
    for i in N:
        m.addConstr(m._u[i] == gp.quicksum(V[i][j] * m._x[j] for j in J))

    objective = kwargs.get('objective', 'utilitarian')
    if objective == 'utilitarian':
        m.setObjective(gp.quicksum(m._u[i] for i in N))
    elif objective == 'maximin':
        z = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY)
        for i in N:
            s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY)
            m.addConstr(m._u[i] - s == z)
        m.setObjective(z)
    else:
        raise Exception('objective {0} not supported'.format(objective))

    m.optimize()
    x_N = {j: m._x[j].X for j in J}
    u_N = {i: m._u[i].X for i in N}

    blocking_IterCount = -1
    eps, S = -1, None

    tf = time.time()
    tt = tf - ts
    out = x_N, u_N, tt, eps, S
    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'wb') as file:
        pickle.dump(out, file)

    blocking_TimeLimit = kwargs.get('blocking_TimeLimit', 60)
    blocking_IterLimit = kwargs.get('blocking_IterLimit', 10)
    blocking_EpsLimit = kwargs.get('blocking_EpsLimit', 0)
    blocking_Starts = {tuple(sorted(N))}

    m.reset()
    for i in N:
        s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY)
        m.addConstr(m._u[i] - s == max(V[i][j]/A[0][j] for j in J))

    m.optimize()
    x_N = {j: m._x[j].X for j in J}
    u_N = {i: m._u[i].X for i in N}

    blocking_IterCount += 1
    eps, S = get_blocking(instance, u_N, TimeLimit=blocking_TimeLimit, blocking_Starts=blocking_Starts)
    blocking_Starts.add(tuple(sorted(S)))

    tf = time.time()
    tt = tf - ts
    out = x_N, u_N, tt, eps, S
    with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'wb') as file:
        pickle.dump(out, file)

    while eps > blocking_EpsLimit and blocking_IterCount <= blocking_IterLimit:

        print('IterCount: {0}'.format(blocking_IterCount))

        for ct, S in enumerate(blocking_Starts):
            intersections = get_intersections(instance, m, u_N, S)
            if intersections is not None:
                print('... Added cut for S no. {0}'.format(ct))
                s = m.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY)
                m.addConstr(gp.quicksum(m.getVarByName(varname)/lam for varname, lam in intersections) - s == 1-1E-6)

        m.reset()
        m.optimize()
        x_N = {j: m._x[j].X for j in J}
        u_N = {i: m._u[i].X for i in N}

        blocking_IterCount += 1
        eps, S = get_blocking(instance, u_N, TimeLimit=blocking_TimeLimit, blocking_Starts=blocking_Starts)
        blocking_Starts.add(tuple(sorted(S)))

        tf = time.time()
        tt = tf - ts
        out = x_N, u_N, tt, eps, S
        with open('{0}/results/solutions/{1}_{2}_{3}.pkl'.format(RELPATH, FILENAME, modelname, blocking_IterCount), 'wb') as file:
            pickle.dump(out, file)

    return out


def get_blocking(instance, u_N, **kwargs):

    N, J, K, A, B, V = instance
    Starts = kwargs.get('blocking_Starts', {tuple(sorted(N))})

    m_S = gp.Model()
    m_S.Params.OutputFlag = kwargs.get('OutputFlag', 1)
    m_S.Params.FeasibilityTol = kwargs.get('FeasibilityTol', 1E-9)
    m_S.Params.MIPFocus = kwargs.get('MIPFocus', 1)
    m_S.Params.NumericFocus = kwargs.get('NumericFocus', 3)
    m_S.Params.TimeLimit = kwargs.get('TimeLimit', 60)
    m_S.NumStart = len(Starts)
    m_S.ModelSense = -1

    m_S._eps = m_S.addVar(vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='eps')
    m_S._y = m_S.addVars(N, vtype=gp.GRB.BINARY, name='y')
    m_S._x = m_S.addVars(J, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='x')
    m_S._u = m_S.addVars(N, vtype=gp.GRB.CONTINUOUS, lb=0, ub=gp.GRB.INFINITY, name='u')

    m_S.setObjective(m_S._eps)

    m_S.addConstr(gp.quicksum(m_S._y[i] for i in N) >= 1)
    for k in K:
        m_S.addConstr(gp.quicksum(A[k][j] * m_S._x[j] for j in J) == gp.quicksum(B[i][k] * m_S._y[i] for i in N))
    for i in N:
        m_S.addConstr(m_S._u[i] == gp.quicksum(V[i][j] * m_S._x[j] for j in J))
    for i in N:
        m_S.addGenConstrIndicator(m_S._y[i], True, m_S._eps - m_S._u[i], gp.GRB.LESS_EQUAL, -u_N[i])

    for StartNumber, Start in enumerate(Starts):
        m_S.Params.StartNumber = StartNumber
        for i in N:
            if i in Start:
                m_S._y[i].Start = 1
            else:
                m_S._y[i].Start = 0

    m_S.optimize()

    eps = m_S._eps.X
    S = {i for i in N if m_S._y[i].X}

    return eps, S


def get_intersections(instance, m, u_N, S, **kwargs):

    N, J, K, A, B, V = instance

    m_S = gp.Model()
    m_S.Params.OutputFlag = kwargs.get('OutputFlag', 1)
    m_S.Params.FeasibilityTol = kwargs.get('FeasibilityTol', 1E-9)
    m_S.Params.NumericFocus = kwargs.get('NumericFocus', 3)
    m_S.Params.OptimalityTol = kwargs.get('OptimalityTol', 1E-9)
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

