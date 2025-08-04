import scipy.io as sio

matfile = r"C:\Users\mahsa\Downloads\ricsimulationironclad\step1\20120803PF_Anesthesia+and+Sleep_George_Toru+Yanagawa_mat_ECoG128\Session1\Condition.mat"
mat = sio.loadmat(matfile)
for k in ['ConditionIndex', 'ConditionTime', 'ConditionLabel']:
    print(f"{k}: {mat[k] if k in mat else 'not found'}")
