import numpy as np
from numpy import linalg as LA

# A set of common functions for the MPS algorithms

def Create_Hamiltonina(Model, S1, Interactions, Fields,d):
    H = np.zeros((d,d,d,d)).astype(np.complex128)

    for j in range(len(Model[0])):
      H += np.einsum("ij,kl->ikjl",Interactions[Model[0][j]]*S1[Model[0][j]],S1[Model[0][j]]) #two-site

    for k in range(len(Model[1])):
      H += np.einsum("ij,kl->ikjl",Fields[Model[1][k]]*S1[Model[1][k]],np.diag(np.ones(2)))    #on-site
      H += np.einsum("ij,kl->ikjl",np.diag(np.ones(2)),Fields[Model[1][k]]*S1[Model[1][k]])
    return H
    
def Create_MPO(Model,S,Interactions,Fields,chi_MPO,data_type,d=2):
    W = np.zeros((chi_MPO,chi_MPO,d,d),dtype=data_type)

    for j in range(2):
        for k in range(chi_MPO-2):
            if j == 0:
              W[j,k+1] = S[Model[0][k]]
            else:
              W[k+1,-1] = Interactions[Model[0][k]]*S[Model[0][k]]
    for m in range(len(Model[1])):
       W[0,-1] +=  Fields[Model[1][m]]*S[Model[1][m]]
    W[0,0] = np.array(np.eye(2),dtype=data_type)
    W[-1,-1] = np.array(np.eye(2),dtype=data_type)
    return W

def Create_random_MPS(L, chis, d=2):
    Gamma,v = [],[[1.]]
    for m in range (L):
       Gamma.append((np.random.rand(chis[m],d,chis[m+1])+1j*np.random.rand(chis[m],d,chis[m+1])))
       norm = np.einsum("aib,aib->",Gamma[m],np.conj(Gamma[m]))
       Gamma[m] = Gamma[m]/np.sqrt(norm)
       v.append(np.random.rand(chis[m+1]))
       v_norm = np.linalg.norm(v[m+1])
       v[m+1] = v[m+1]/v_norm
    return Gamma,v

def Convert_to_A_and_B(Gamma, v, Le):
  A,B = [],[]
  B.append([1.])
  for j in range(Le):
    A.append(np.einsum("a,aib->aib",v[j],Gamma[j]))
    B.append(np.einsum("aib,b->aib",Gamma[j],v[j+1])) 
  A.append(np.array([1.]))
  return A,B


def ka(L , cap, exact=False):
  if exact:
     k=np.zeros(L+1,dtype=int)
     for j in range(L+1):
      if j<L/2:
       k[j]=int(pow(2,j))
      else:
       k[j]=int(pow(2,L-j))
     return k
  else:
     k=np.zeros(L+1,dtype=int)
     for j in range(L+1):
      if j<L/2:
       k[j]=min(int(pow(2,j)),cap)
      else:
       k[j]=min(int(pow(2,L-j)),cap)
     return k

def i_trunc(chi,Fs,d=2):
  S,V,D = LA.svd(Fs,full_matrices=False)
    
  lam = V[:chi]/np.linalg.norm(V[:chi])

  A=S[:,:chi].reshape(chi,d,chi)
  B=D[:chi].reshape(chi,d,chi)
  
  return A,lam,B

def Pauli():
  S_z = np.diag([1.,-1.]).astype(np.complex128)
  S_x =np.array([[0.,1.],[1.,0.]],dtype = np.complex128)
  S_y=np.array([[0.,complex(0,-1.)],[complex(0,1.),0.]],dtype = np.complex128) #Pauli Matrices
  S1 = {
     "x": S_x, 
     "y": S_y,
     "z": S_z}
  return S_z, S_x, S_y, S1

def Model_coefficients(J_xx,J_yy,J_zz,h_x,h_y,h_z):
  Interactions = {
                  "x":J_xx,
                  "y":J_yy,
                  "z":J_zz}
  Fields = {
          "x":h_x,  #Fields
          "y":h_y,
          "z":h_z}
  return Interactions,Fields

def truncate(j,Theta,k,L,d=2):
    S,V,D=np.linalg.svd(Theta,full_matrices=False)
    
    lam = V[:k[j+1]]/LA.norm(V[:k[j+1]])
   
    A=S[:,:k[j+1]].reshape(k[j],d,k[j+1])
    if j!=L-1:
     B=D[:k[j+1]].reshape(k[j+1],d,-1)
    else:
     B=np.array([[[1.]]])
    return A,lam,B
