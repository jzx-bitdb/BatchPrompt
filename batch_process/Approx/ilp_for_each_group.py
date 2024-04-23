import gurobipy as gp
from gurobipy import GRB
import numpy as np
import time
import json
def solve_MIQCP_gurobi_quicksum(n, m, M, grpAff, grpDemAff, queCost, demCost, tCost, k0, k1, k2):
    model = gp.Model("ILP")
    
    x = model.addMVar((n, n), vtype=GRB.BINARY, name="x")
    y = model.addMVar((n, m), vtype=GRB.BINARY, name="y")
    z = model.addMVar((n, n, m), vtype=GRB.BINARY, name="z")
    u = model.addMVar(n, vtype=GRB.BINARY, name="u")
    model.addConstrs(gp.quicksum(x[j, i] for j in range(n)) == 1 for i in range(n))
    for j in range(n):
        for i in range(n):
            for i_star in range(n):
                if i_star<i:
                    model.addConstr(x[j,i]*x[j,i_star] <= grpAff[i][i_star])
    for i in range(n):
        model.addConstr(gp.quicksum(grpDemAff[i][l]*x[j,i]*y[j,l] for j in range(n) for l in range(m)) >= 1)
    for j in range(n):
        model.addConstr(gp.quicksum(x[j,i]*queCost[i] for i in range(n))+gp.quicksum(y[j,l]*demCost[l] for l in range(m))+u[j]*tCost<=k2)
    for j in range(n):
        model.addConstr(u[j]-(gp.quicksum(x[j,i] for i in range(n)) + gp.quicksum(y[j,l] for l in range(m))) <= 0)
        model.addConstr(M*u[j]-(gp.quicksum(x[j,i] for i in range(n)) + gp.quicksum(y[j,l] for l in range(m)))>=0)
    
    model.setObjective(gp.quicksum(x[j,i]*queCost[i] for j in range(n) for i in range(n)) + gp.quicksum(y[j,l]*demCost[l] for j in range(n) for l in range(m)) + gp.quicksum(u[j]*tCost for j in range(n)), sense=GRB.MINIMIZE)
    
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("Objective value: {}".format(model.ObjVal))
    print(y[0][0].X)
    groups = {}
    group_num = 0
    flag = 0
    for i in range(n):
        flag = 0
        for j in range(n):
            if x[i][j].X != 0:
                if flag == 0:
                    group_num += 1
                    flag = 1
                    groups[group_num] = []
                groups[group_num].append(j)
    demos = {}
    group_num = 0
    flag = 0
    for i in range(n):
        flag = 0
        for j in range(m):
            if y[i][j].X != 0:
                if flag == 0:
                    group_num += 1
                    flag = 1
                    demos[group_num] = []
                demos[group_num].append(j)
    print(groups)
    print(demos)
    with open('groups.json', 'w') as f:
        f.write(json.dumps(groups, indent=4))
    with open('demos.json', 'w') as f:
        f.write(json.dumps(demos, indent=4))
    print(np.mean([len(groups[key]) for key in groups]))
    print(np.mean([len(demos[key]) for key in demos]))
    model.write("lp_ex_without96.sol")
   
def solve_MIQCP_gurobi_quicksum_without_groupaff(n, m, M, grpAff, grpDemAff, queCost, demCost, tCost, k0, k1, k2):
    model = gp.Model("ILP")
    
    x = model.addMVar((n, n), vtype=GRB.BINARY, name="x")
    y = model.addMVar((n, m), vtype=GRB.BINARY, name="y")
    z = model.addMVar((n, n, m), vtype=GRB.BINARY, name="z")
    u = model.addMVar(n, vtype=GRB.BINARY, name="u")
    model.addConstrs(gp.quicksum(x[j, i] for j in range(n)) == 1 for i in range(n))
    for i in range(n):
        model.addConstr(gp.quicksum(grpDemAff[i][l]*x[j,i]*y[j,l] for j in range(n) for l in range(m)) >= 1)
    for j in range(n):
        model.addConstr(gp.quicksum(x[j,i]*queCost[i] for i in range(n))+gp.quicksum(y[j,l]*demCost[l] for l in range(m))+u[j]*tCost<=k2)
    for j in range(n):
        model.addConstr(u[j]-(gp.quicksum(x[j,i] for i in range(n)) + gp.quicksum(y[j,l] for l in range(m))) <= 0)
        model.addConstr(M*u[j]-(gp.quicksum(x[j,i] for i in range(n)) + gp.quicksum(y[j,l] for l in range(m)))>=0)
    
    model.setObjective(gp.quicksum(x[j,i]*queCost[i] for j in range(n) for i in range(n)) + gp.quicksum(y[j,l]*demCost[l] for j in range(n) for l in range(m)) + gp.quicksum(u[j]*tCost for j in range(n)), sense=GRB.MINIMIZE)
    
    model.optimize()
    if model.status == GRB.OPTIMAL:
        print("Objective value: {}".format(model.ObjVal))
    print(y[0][0].X)
    groups = {}
    group_num = 0
    flag = 0
    for i in range(n):
        flag = 0
        for j in range(n):
            if x[i][j].X != 0:
                if flag == 0:
                    group_num += 1
                    flag = 1
                    groups[group_num] = []
                groups[group_num].append(j)
    demos = {}
    group_num = 0
    flag = 0
    for i in range(n):
        flag = 0
        for j in range(m):
            if y[i][j].X != 0:
                if flag == 0:
                    group_num += 1
                    flag = 1
                    demos[group_num] = []
                demos[group_num].append(j)
    print(groups)
    print(demos)
    with open('groups.json', 'w') as f:
        f.write(json.dumps(groups, indent=4))
    with open('demos.json', 'w') as f:
        f.write(json.dumps(demos, indent=4))
    print(np.mean([len(groups[key]) for key in groups]))
    print(np.mean([len(demos[key]) for key in demos]))
    model.write("lp_ex_without96.sol")