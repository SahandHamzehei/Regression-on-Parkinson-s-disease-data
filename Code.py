# %% Import Libraries

from sklearn.metrics import r2_score 
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# %% Algorithms Defination

class SolvMinProbl():
    def __init__(self, y, A, y_test, A_test, y_val, A_val, mean, var):
        
        self.matr=A
        self.Np=A.shape[0]
        self.Nf=A.shape[1]
        
        self.matr_test=A_test
        self.matr_val=A_val

        self.mean = mean
        self.var = var

        self.vect=y
        self.vect_test=y_test
        self.vect_val=y_val
        
        self.sol=np.zeros((self.Nf, 1), dtype=float)
        self.err=0
        self.mse=0
        self.stat=0
        
    def MSE_un(self):
        self.mse=np.zeros((3, 1), dtype=float)
        
        y_train_estimated = self.var * np.dot(self.matr,self.sol) + self.mean
        y_train = self.var * self.vect + self.mean
        
        y_validation = self.var * self.vect_val + self.mean
        y_val_estimated = self.var * np.dot(self.matr_val,self.sol) + self.mean
        
        ytest = self.var * self.vect_test + self.mean
        y_test_estimated = self.var * np.dot(self.matr_test,self.sol) + self.mean
        
        self.mse[0] = (np.linalg.norm(y_train - y_train_estimated)**2)/self.matr.shape[0]
        self.mse[1] = (np.linalg.norm(y_validation - y_val_estimated)**2)/self.matr_val.shape[0]
        self.mse[2] = (np.linalg.norm(ytest - y_test_estimated)**2)/self.matr_test.shape[0]
        
        return self.mse[0], self.mse[1], self.mse[2]
        
    def stat_u(self):
        self.stat=np.zeros((3, 2), dtype=float)
        y_train_estimated = self.var * np.dot(self.matr,self.sol) + self.mean
        y_train = self.var * self.vect + self.mean
        errtr= y_train - y_train_estimated
        self.stat[0][0]=errtr.mean()
        self.stat[0][1]=errtr.std()
        
        y_validation = self.var * self.vect_val + self.mean
        y_val_estimated = self.var * np.dot(self.matr_val,self.sol) + self.mean
        errva= y_validation - y_val_estimated
        self.stat[1][0]=errva.mean()
        self.stat[1][1]=errva.std()
        
        ytest = self.var * self.vect_test + self.mean
        y_test_estimated = self.var * np.dot(self.matr_test,self.sol) + self.mean
        errte= ytest- y_test_estimated
        self.stat[2][0]=errte.mean()
        self.stat[2][1]=errte.std()
        
        return self.stat
 
    def determ_coeff(self):
        ytest = self.var * self.vect_test + self.mean
        y_test_estimated = self.var * np.dot(self.matr_test,self.sol) + self.mean

        r2 = r2_score(ytest , y_test_estimated)
        return r2
    
# %% Plotting    

    # ======================= Plot Weight Vector =======================
    
    def plot_w(self, title ="Solution"):
        
        w = self.sol
        n = np.arange(self.Nf)
         
        features=['age','sex','motor_UPDRS','Jitter(%)','Jitter(Abs)',
                  'Jitter:RAP','Jitter:PPQ5','Jitter:DDP','Shimmer',
                  'Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11',
                  'Shimmer:DDA','NHR','HNR','RPDE','DFA','PPE']
        
        plt.figure(tight_layout=True, dpi = 300)
        plt.plot(n, w, 'g-o')
        plt.xticks(n, features, rotation = 90)
        plt.xticks(np.arange(0,19,1))
        plt.ylabel('Weight Vector')        
        plt.margins(0.01,0.1)        
        plt.title(title)
        plt.grid()
        plt.show()
        
    # =================== y test estimation VS y test ===================
    def comparison(self,title, labelx, labely, y, A, mean , var):
        w = self.sol
        
        
        y_est = (np.dot(A, w) * var)+mean
        y = (y * var) + mean #un-normalized y
        
        plt.figure(dpi = 300)
        plt.plot(np.linspace(0, 60), np.linspace(0, 60), 'lightcoral')
        plt.scatter(y,y_est, s=5, color='seagreen')
        plt.xlabel(labelx)
        plt.ylabel(labely)
        plt.title(title)
        plt.grid()
        
    # =========================== Histogram ===========================
    def hist(self, y, A, y_test, A_test, mean , var, title ):
        w = self.sol
        
        
        y_tr = (y * var) + mean
        y_est_tr = (np.dot(A, w) * var) + mean
        y_test = (y_test * var) + mean 
        y_est_test = (np.dot(A_test, w) * var) + mean
        
        plt.figure(dpi = 300)
        plt.hist(y_tr-y_est_tr, bins = 75, density = True, histtype = 'bar', label = "training", color='lightcoral')
        plt.hist(y_test-y_est_test, bins = 75, density = True, histtype = 'bar', label = "test", color='seagreen')
        plt.xlabel('Error on estimation')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend(loc = 'best')
        plt.show()
    
    
    # =================== Histogram with Validation ==================
    def hist_plus_val(self, y, A, y_test, A_test, y_val, A_val, mean , var, title ):
        w = self.sol
        
        
        y_tr = (y * var) + mean
        y_est_tr = (np.dot(A, w) * var) + mean
        
        y_test = (y_test * var) + mean
        y_est_test = (np.dot(A_test, w) * var)+mean        
        
        y_val = (y_val * var) + mean
        y_est_val = (np.dot(A_val, w) * var)+mean         
        
        plt.figure(dpi = 300)
        plt.hist(y_tr-y_est_tr, bins = 50, density = True, histtype = 'bar', label = "training", color='lightcoral')
        plt.hist(y_test-y_est_test, bins = 50, density = True, histtype = 'bar', label = "test", color='seagreen')
        plt.hist(y_val-y_est_val, bins = 50, density = True, histtype = 'bar', label="validation", color='wheat')
        
        plt.xlabel('Error on estimation')
        plt.ylabel('Frequency')
        plt.title(title)
        plt.legend(loc = 'best')
        plt.show()
        
    # ========================= Plot Error ========================
    def plot_err(self, title = 'Square error', logy=1, logx=0):
        err = self.err
        plt.figure(dpi = 300)
        
        if(logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 1], label='train')
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 1], label='train')
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 1], label='train')
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 1], label='train')
            
            
        if (logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 2], label='val')
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 2], label='val')
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 2], label='val')
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 2], label='val')
            
        if(logy == 0) & (logx == 0):
            plt.plot(err[:, 0], err[:, 3], label='test')
        if (logy == 1) & (logx == 0):
            plt.semilogy(err[:, 0], err[:, 3], label='test')
        if (logy == 0) & (logx == 1):
            plt.semilogx(err[:, 0], err[:, 3], label='test')
        if (logy == 1) & (logx == 1):
            plt.loglog(err[:, 0], err[:, 3], label='test')
            
        if (logy == 1) & (logx == 1):
            plt.semilogy(errtrain, color='tab:blue')
            plt.semilogy(errtest, color='tab:green')
            plt.semilogy(errval, color='tab:red', linestyle=':')
            
        plt.xlabel('n')
        plt.ylabel('e(n)')
        plt.minorticks_on()
        plt.legend(['Training set', 'Validation set', 'Testing set'])
        plt.title(title)
        plt.margins(0.01, 0.1)
        plt.grid()
        plt.show()
        
    


# %% Methods

    # ========================= Linear Least Square ========================
class LLS(SolvMinProbl):
    def run(self):
        A=self.matr
        A_test=self.matr_test
        
        y=self.vect
        y_test=self.vect_test
      
        w = np.random.rand(self.Nf, 1)
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)), A.T), y)
        
        self.sol = w
        self.err = np.zeros((1, 4), dtype=float)
        self.err[0, 1] = np.linalg.norm(y-np.dot(A, w))**2/A.shape[0]
        self.err[0, 3] = np.linalg.norm(y_test-np.dot(A_test, w)) ** 2 / A_test.shape[0]
        
        return self.err[0, 1], self.err[0, 3]
    

class CONJ(SolvMinProbl):
    def run(self):
        A = self.matr
        y = self.vect
        
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
        
        w = np.zeros((self.Nf, 1), dtype=float)
        self.err = np.zeros((self.Nf, 4), dtype=float)
        
        it = 0
        Q = 2 * np.dot(A.T, A)
        b = 2 * np.dot(A.T, y)
        g = -b
        d = -g
        
        for it in range(self.Nf):
            a = -1 * np.dot(d.T, g)/np.dot(np.dot(d.T, Q), d)
            w = w + a * d
            g = g + a * np.dot(Q, d)
            beta = np.dot(np.dot(g.T, Q), d)/np.dot(np.dot(d.T, Q), d)
            d = -1 * g + beta*d
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(y-np.dot(A, w) ) ** 2 / A.shape[0]
            self.err[it, 2] = np.linalg.norm(y_val-np.dot(A_val, w) ) ** 2 / A_val.shape[0]
            self.err[it, 3] = np.linalg.norm(y_test-np.dot(A_test, w) ) ** 2 / A_test.shape[0]
        self.sol = w
        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]

class SGDA(SolvMinProbl):
    def run(self, y_val, A_val, y_test, A_test, gamma, Beta1, Beta2, it, epsilon):
        A = self.matr
        y = self.vect
        
        w = np.random.rand(self.Nf,1)
        self.MSE = np.zeros((it,4),dtype=float)
        self.err=np.zeros((it,4),dtype=float)
        
        mu1 = 0
        mu2 = 0
        mu1_hat = 0
        mu2_hat = 0
        stop = 0
        
        for t in range(it):
            if(stop<50):
                grad = 2*np.dot(A.T,(np.dot(A,w)-y))
                mu1 = mu1*Beta1+(1-Beta1)*grad
                mu2 = mu2*Beta2+(1-Beta2)*pow(grad,2)
                mu1_hat = mu1/(1-pow(Beta1,(t+1)))
                mu2_hat = mu2/(1-pow(Beta2,(t+1)))
                w = w - gamma * mu1_hat/(np.sqrt(mu2_hat)+epsilon)
                
                self.MSE[t,0]=t
                self.MSE[t,1]=np.linalg.norm(y-np.dot(A,w))**2/A.shape[0]
                self.MSE[t,2] = np.linalg.norm(y_val-np.dot(A_val,w))**2/A_val.shape[0]
                self.MSE[t,3] = np.linalg.norm(y_test-np.dot(A_test,w))**2/A_test.shape[0]
                
                if (self.MSE[t,2]>self.MSE[t-1,2]):
                    stop += 1
                else:
                    stop = 0
            else:
                print('The number of Nit is :',t)
                break
        
        self.sol = w
        self.err = self.MSE[0:t]

        return self.err[-1,1], self.err[-1,2], self.err[-1,3]
            
class SolveRidge(SolvMinProbl):
    def run(self):
        A = self.matr
        y = self.vect
        A_val = self.matr_val
        y_val = self.vect_val
        A_test = self.matr_test
        y_test = self.vect_test
        I = np.eye(self.Nf)
        lamb = 300
        self.err = np.zeros((lamb, 4), dtype=float)
        for it in range(lamb):
            w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A)+it*I), A.T), y)
            self.err[it, 0] = it
            self.err[it, 1] = np.linalg.norm(y-np.dot(A, w)) ** 2 / A.shape[0]
            self.err[it, 2] = np.linalg.norm(y_val-np.dot(A_val, w) ) ** 2 / A_val.shape[0]
            # print(self.err[it, 2])
            self.err[it, 3] = np.linalg.norm(y_test-np.dot(A_test, w) ) ** 2 / A_test.shape[0]
        best_lamb = np.argmin(self.err[:, 2])
        w = np.dot(np.dot(np.linalg.inv(np.dot(A.T, A) + best_lamb * I), A.T), y)
        self.sol = w
        err = self.err
        print("best lambda is :", best_lamb)

        #plotting the MSE with respect to lambda 
        plt.figure(dpi = 300)
        plt.semilogy(err[:, 0], err[:, 1], label='train')
        plt.semilogy(err[:, 0], err[:, 2], label='val')
        plt.xlabel('lambda')
        plt.ylabel('error')
        plt.legend()
        plt.title('MSE')
        plt.grid()
        plt.show()
        
        return self.err[-1, 1], self.err[-1, 2], self.err[-1, 3]


# %% Reade Data-set

plt.close('all')
x = pd.read_csv("parkinsons_updrs.csv")
x.info()

features = list(x.columns)
xnorm = (x-x.mean()) / x.std()
cc = xnorm.cov()

# %% Plot Correlation

X = x.drop(['subject#','test_time'],axis=1)
features_droped = list(X.columns)

Xnorm=(X-X.mean())/X.std()
c = Xnorm.cov()

plt.figure(figsize=(10,10), dpi = 300)
plt.matshow(np.abs(c.values),fignum=0)
plt.xticks(np.arange(len(features_droped)), features_droped, rotation=90)
plt.yticks(np.arange(len(features_droped)), features_droped, rotation=0)    
plt.colorbar()


# %% Prepare data

features_to_drop=['Shimmer','Shimmer(dB)','Shimmer:APQ3','Shimmer:APQ5','Shimmer:APQ11','Shimmer:DDA','NHR','HNR','RPDE','DFA','PPE']
x1=x.copy(deep=True)
x1=x1.drop(features_to_drop, axis=1)
xnorm1=(x1-x1.mean())/x1.std()
c1=xnorm1.cov()

f = ['subject#','test_time']
x2 = x.copy(deep=True)
x2 = x2.drop(f, axis=1)

# %% Main Part
np.random.seed(11)

data = x2.sample(frac=1).reset_index(drop=True)

Np=data.shape[0] # number of patients 
Nf=data.shape[1] # number of features

data_train = data[0:int(Np/2)]
data_val = data[int(Np/2):int(Np*0.75)]
data_test = data[int(Np*0.75):Np]

mean = np.mean(data_train.values, 0)
std = np.std(data_train.values, 0)

# %% Standardizing

data_train_nonorm=data_train.values
data_train_norm = (data_train.values - mean)/std

data_val_nonorm=data_val.values
data_val_norm = (data_val.values - mean)/std

data_test_nonorm=data_test.values
data_test_norm = (data_test.values - mean)/std 


F=3
y_train = data_train_norm[:, F]
y_trr=np.reshape(y_train,(2937,1))
x_train = np.delete(data_train_norm, F, 1)  #in this way i'm removing the total_updrs column

y_test = data_test_norm[:, F]
y_txt=np.reshape(y_test,(1469,1))
x_test = np.delete(data_test_norm, F, 1)  #in this way i'm removing the total_updrs column

y_val = data_val_norm[:, F]
y_valid=np.reshape(y_val,(1469,1))
x_val = np.delete(data_val_norm, F, 1)  #in this way i'm removing the total_updrs column

# Now data is ready

# %% Initialization

logx = 0
logy = 1

lls_stat=np.zeros((3, 2), dtype=float)
conj_stat=np.zeros((3, 2), dtype=float)
sgwa_stat=np.zeros((3, 2), dtype=float)
ridge_stat=np.zeros((3, 2), dtype=float)

R2=np.zeros((4,1), dtype=float)

mat= np.zeros((3, 4), dtype=float) 

mse_train = np.zeros((4, 1), dtype=float)   
mse_test = np.zeros((4, 1), dtype=float)
mse_val = np.zeros((4, 1), dtype=float)


# %% LLS

m = LLS(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F],std[F])
mse_train[0], mse_test[0] = m.run()
mat[:,0]=m.MSE_un()
lls_stat=m.stat_u()
m.plot_w('Optimized weights - Linear Least Squares')
m.comparison('y estimation train vs y train (LLS)','y_est_train','y_train', y_train, x_train, mean[F],std[F])  #sto estraendo la media e la varianza della colonna total updrs
m.comparison('y estimation test vs y test (LLS)','y_est_test','y_test', y_test, x_test, mean[F],std[F])  #sto estraendo la media e la varianza della colonna total updrs
# m.hist( y_train, x_train, y_test, x_test,mean[F],std[F], 'Error Histograms (LLS)')
m.hist_plus_val(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F],std[F], 'Error Histograms (LLS)')
R2[0]=m.determ_coeff()

# %% CONJUGATE

c= CONJ(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F],std[F])
mse_train[1], mse_val[1], mse_test[1] =c.run()
mat[:,1]=c.MSE_un()

c.plot_w('Optimum Weight Vector - Conjugate')
c.comparison('y estimation train vs y train (Conjugate)','y_est_train','y_train', y_trr, x_train, mean[F],std[F])
c.comparison('y estimation test vs y test (Conjugate)','y_est_test','y_test', y_test, x_test, mean[F],std[F])
c.hist_plus_val(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F],std[F], 'Error Histograms (Conjugate)' )
c.plot_err('Conjugate : square error', logy, logx)
conj_stat=c.stat_u()
R2[1]=c.determ_coeff()

# %% STOCHASTIC GRADIENT WITH ADAM

s= SGDA(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F],std[F])
mse_train[2], mse_val[2], mse_test[2]=s.run( y_valid, x_val, y_txt, x_test, 0.001, 0.99, 0.999, 20000, 1e-8)
mat[:,2]=s.MSE_un()

s.plot_w('Optimum Weight Vector - Stochastic gradient with Adam')
s.plot_err('square error - Stochastic gradient with Adam', logy, logx)
s.comparison('y estimation train vs y train Stochastic gradient with Adam','y_est_train','y_train', y_trr, x_train, mean[F],std[F])
s.comparison('y estimation test vs y test Stochastic gradient with Adam','y_est_test','y_test', y_test, x_test, mean[F],std[F])
s.hist_plus_val(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F],std[F], 'Error Histograms (Stochastic gradient with Adam)' )
sgwa_stat=s.stat_u()
R2[2]=s.determ_coeff()

# %% RIDGE

r= SolveRidge(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F],std[F])
mse_train[3], mse_val[3], mse_test[3]=r.run()
mat[:,3]=r.MSE_un()
r.plot_w('Optimum Weight Vector - Ridge Regression')
r.comparison('y estimation train vs y train - Ridge Regression','y_est_train','y_train', y_trr, x_train, mean[F],std[F])
r.comparison('y estimation test vs y test - Ridge Regression ','y_est_test','y_test', y_test, x_test, mean[F],std[F])
r.hist_plus_val(y_trr, x_train, y_txt, x_test, y_valid, x_val, mean[F],std[F], 'Ridge Regression' )
R2[3]=r.determ_coeff()
ridge_stat=r.stat_u()


# %% Determination coefficient 

plt.figure(dpi = 300)
Algorithms = ['LLS', 'CA', 'SGwA', 'Ridge']
Results = [0.904409, 0.904408, 0.904410, 0.904354]
norm = [float(i) ** 1000 for i in Results]
x_pos = [i for i, _ in enumerate(Algorithms)]

plt.bar(x_pos, norm, width=0.4, color=(0.2, 0.4, 0.6, 0.6))
plt.xlabel("Algorithms")
plt.ylabel("Results")
plt.title("Determination coefficient Results")
plt.xticks(x_pos, Algorithms)
plt.show()

