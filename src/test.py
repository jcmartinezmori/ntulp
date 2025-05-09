
N, J, K, A, B, V
u_N

Starts = set()
for j in J:

    eps_j = 0
    S_j = None

    A_j = A[0][j]
    N_j_V_j = sorted([(i, V[i][j]) for i in N],  key=lambda x: x[1], reverse=True)
    for ct in range(1, len(N) + 1):
        if N_j_V_j[ct - 1][1] <= 0:
            break
        eps = min(V_ij * ct / A_j - u_N[i] for i, V_ij in N_j_V_j[:ct])
        if eps > eps_j:
            eps_j = eps
            S_j = tuple(sorted(i for i, _ in N_j_V_j[:ct]))

    if S_j is not None:
        Starts.add(S_j)
