import wandb
import numpy as np
class Optimizers:
    def optimize(self,FWNN):
        if FWNN.optimizer=="sgd":
            epochs=FWNN.epochs
            L=FWNN.noOfHL+1
            k=10
            x_train=FWNN.x_train
            y_train=FWNN.y_train
            eta=FWNN.learningRate
            batchSize=FWNN.batchSize
            deltaw=[]
            deltab=[]
            loss=[]
            trainingLoss=[]
            validationLoss=[]
            trainingaccuracy=[]
            validationaccuracy=[]
            for epoch in range(epochs):
                loss=[]
                for i in range(x_train.shape[0]):
                    if(i%batchSize==0):
                        if(i!=0):
                            #update gradient with L2 regularization term
                            deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                            #update the weights and biases
                            FWNN.W[1:] = [FWNN.W[i] - eta * deltaw[i]/batchSize for i in range(1, L+1)]
                            FWNN.b[1:] = [FWNN.b[i] - eta * deltab[i]/batchSize for i in range(1, L+1)]
                        #reinitialize the gradient after a batch
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=w_g
                        deltab=b_g
                    else:
                        #accumulates the gradient normally
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=FWNN.acc_grad(deltaw,w_g)
                        deltab=FWNN.acc_grad(deltab,b_g)
                    
                    #append loss for this datapoint
                    loss.append(FWNN.calculateLoss(yhat,i))

                #update gradient with L2 regularization term
                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                #update the weights and biases
                FWNN.W[1:] = [FWNN.W[i] - eta * deltaw[i] for i in range(1, L+1)]    #because of 1 indexing starting list update from 1
                FWNN.b[1:] = [FWNN.b[i] - eta * deltab[i] for i in range(1, L+1)]
                #append the training Loss of this epoch in the list with L2 regularization term
                trainingLoss.append((np.sum(loss)+FWNN.L2regularization())/FWNN.x_train.shape[0])
                #append the validation Loss of this epoch in the list with L2 regularization term
                validationLoss.append((FWNN.calculateValidationLoss()+FWNN.L2regularization())/FWNN.x_valid.shape[0]);
                #calculates the training accuracy after the epoch
                accuracytrain=FWNN.calculateAccuracy("train")
                trainingaccuracy.append(accuracytrain)
                #calculates the validation accuracy after the epoch
                accuracyvalid=FWNN.calculateAccuracy("validation")
                validationaccuracy.append(accuracyvalid)
                #wandb logs 
                wandb.log({'trainingloss':trainingLoss[epoch],'validationloss':validationLoss[epoch], 'trainingaccuracy':accuracytrain, 'validationaccuracy':accuracyvalid,'epoch':epoch })
                print("At epoch:{} trainingloss: {} Training accuracy: {} validation accuracy: {}".format(epoch,trainingLoss[epoch],accuracytrain,accuracyvalid))
            #returns the training and test accuracy of the model after the model is trained on all epochs
            accuracytrain=FWNN.calculateAccuracy("train")
            accuracytest=FWNN.calculateAccuracy("test")
            return trainingLoss,accuracytrain,accuracytest
        if FWNN.optimizer=="mgd":
            Gamma=FWNN.gamma
            epochs=FWNN.epochs
            L=FWNN.noOfHL+1
            k=10
            x_train=FWNN.x_train
            y_train=FWNN.y_train
            eta=FWNN.learningRate
            batchSize=FWNN.batchSize
            deltaw=[]
            deltab=[]
            prev_w=[0]*(L+1)
            prev_b=[0]*(L+1)
            loss=[]
            trainingLoss=[]
            trainingaccuracy=[]
            validationaccuracy=[]
            validationLoss=[]
            t=0
            for epoch in range(epochs):
                loss=[]
                for i in range(x_train.shape[0]):
                    if(i%batchSize==0):
                        if(i!=0):
                            #if this is the first udpate then prev_w is not yet initialised
                            if t==0:
                                #update gradient with L2 regularization term
                                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                                #update the weights and biases
                                FWNN.W[1:] = [FWNN.W[i] - eta * deltaw[i]/batchSize for i in range(1, L+1)]
                                FWNN.b[1:] = [FWNN.b[i] - eta * deltab[i]/batchSize for i in range(1, L+1)]
                                #calculate history for next batch
                                prev_w[1:]=[eta*deltaw[i]/batchSize for i in range(1, L+1)]
                                prev_b[1:]=[eta*deltab[i]/batchSize for i in range(1, L+1)]
                                t=1
                            else:
                                #update gradient with L2 regularization term
                                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                                #update the weights and biases
                                FWNN.W[1:] = [FWNN.W[i] -  Gamma * prev_w[i] - eta * deltaw[i]/batchSize for i in range(1, L+1)]
                                FWNN.b[1:] = [FWNN.b[i] -  Gamma * prev_b[i] - eta * deltab[i]/batchSize for i in range(1, L+1)]
                                #calculate history for next batch
                                prev_w[1:]=[Gamma * prev_w[i] + eta*deltaw[i]/batchSize for i in range(1, L+1)]
                                prev_b[1:]=[Gamma * prev_b[i] + eta*deltab[i]/batchSize for i in range(1, L+1)]
                        #reinitialize the gradient after a batch        
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=w_g
                        deltab=b_g
                    else:
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=FWNN.acc_grad(deltaw,w_g)
                        deltab=FWNN.acc_grad(deltab,b_g)
                        
                    #append loss for this datapoint
                    loss.append(FWNN.calculateLoss(yhat,i))
                #update gradient with L2 regularization term
                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                #update the weights and biases
                FWNN.W[1:] = [FWNN.W[i] - Gamma * prev_w[i] - eta * deltaw[i]/batchSize for i in range(1, L+1)]
                FWNN.b[1:] = [FWNN.b[i] - Gamma * prev_b[i] - eta * deltab[i]/batchSize for i in range(1, L+1)]
                #append the training Loss of this epoch in the list with L2 regularization term
                trainingLoss.append((np.sum(loss)+FWNN.L2regularization())/FWNN.x_train.shape[0])
                #append the validation Loss of this epoch in the list with L2 regularization term
                validationLoss.append((FWNN.calculateValidationLoss()+FWNN.L2regularization())/FWNN.x_valid.shape[0]);
                #calculates the training accuracy after the epoch
                accuracytrain=FWNN.calculateAccuracy("train")
                trainingaccuracy.append(accuracytrain)
                #calculates the validation accuracy after the epoch
                accuracyvalid=FWNN.calculateAccuracy("validation")
                validationaccuracy.append(accuracyvalid)
                #wandb logs
                wandb.log({'trainingloss':trainingLoss[epoch],'validationloss':validationLoss[epoch], 'trainingaccuracy':accuracytrain, 'validationaccuracy':accuracyvalid,'epoch':epoch })
                print("At epoch:{} trainingloss: {} Training accuracy: {} validation accuracy: {}".format(epoch,trainingLoss[epoch],accuracytrain,accuracyvalid))
            #returns the training and test accuracy of the model after the model is trained on all epochs 
            accuracytrain=FWNN.calculateAccuracy("train")
            accuracytest=FWNN.calculateAccuracy("test")
            return trainingLoss,accuracytrain,accuracytest
        if FWNN.optimizer=="nag":
            Gamma=FWNN.gamma
            epochs=FWNN.epochs
            L=FWNN.noOfHL+1
            k=10
            x_train=FWNN.x_train
            y_train=FWNN.y_train
            eta=FWNN.learningRate
            batchSize=FWNN.batchSize
            deltaw=[]
            deltab=[]
            prev_w=[0]*(L+1)
            prev_b=[0]*(L+1)
            v_w=[0]*(L+1)
            v_b=[0]*(L+1)
            prev_w[1:]=[FWNN.W[i] - FWNN.W[i] for i in range(1, L+1)]
            prev_b[1:]=[FWNN.b[i] - FWNN.b[i] for i in range(1, L+1)]
            W_l=FWNN.W
            b_l=FWNN.b
            trainingLoss=[]
            trainingaccuracy=[]
            validationaccuracy=[]
            validationLoss=[]
            t=0
            # In this implementation the W_l and b_l are the current weights and the FWNN.W and FWNN.b will act as look ahead weights
            for epoch in range(epochs):
                loss=[]
                for i in range(x_train.shape[0]):
                    if i!=0:
                        #reassigning FWNN.W and FWNN.b after loss calculation
                        FWNN.W=temp_w  
                        FWNN.b=temp_b
                    if i%batchSize==0:
                        if t==0:
                            #calculating lookahead after first batch
                            FWNN.W[1:] = [FWNN.W[i] - Gamma * prev_w[i] for i in range(1, L+1)]
                            FWNN.b[1:] = [FWNN.b[i] - Gamma * prev_b[i] for i in range(1, L+1)]
                            t=1
                        else:
                            #update w gradient with L2 regularization term
                            deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                            #calculating the history
                            v_w[1:] = [Gamma * prev_w[i] + eta * deltaw[i]/batchSize for i in range(1, L+1)]
                            v_b[1:] = [Gamma * prev_b[i] + eta * deltab[i]/batchSize for i in range(1, L+1)]
                            #Update real W and bs
                            W_l[1:] = [W_l[i] - v_w[i] for i in range(1, L+1)]
                            b_l[1:] = [b_l[i] - v_b[i] for i in range(1, L+1)]
                            prev_w=v_w
                            prev_b=v_b
                            #calculating look aheads for the next batch
                            FWNN.W[1:] = [FWNN.W[i] - v_w[i] for i in range(1, L+1)]
                            FWNN.b[1:] = [FWNN.b[i] - v_b[i] for i in range(1, L+1)]
                        #reinitialize the gradient after a batch    
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=w_g
                        deltab=b_g    
                    else:
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=FWNN.acc_grad(deltaw,w_g)
                        deltab=FWNN.acc_grad(deltab,b_g)
                    #storing the real weights and bias in a temp_w, temp_b
                    temp_w=W_l
                    temp_b=b_l
                    #assinging real weights and bias to FWNN.W and FWNN.b to calculate the loss
                    FWNN.W=W_l
                    FWNN.b=b_l
                    #append loss for this datapoint
                    loss.append(FWNN.calculateLoss(yhat,i))
                #update gradient with L2 regularization term
                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                v_w[1:] = [Gamma * prev_w[i] + eta * deltaw[i]/batchSize for i in range(1, L+1)]
                v_b[1:] = [Gamma * prev_b[i] + eta * deltab[i]/batchSize for i in range(1, L+1)]
                W_l[1:] = [W_l[i] - v_w[i] for i in range(1, L+1)]
                b_l[1:] = [b_l[i] - v_b[i] for i in range(1, L+1)]
                #append the training Loss of this epoch in the list with L2 regularization term
                trainingLoss.append((np.sum(loss)+FWNN.L2regularization())/FWNN.x_train.shape[0])
                #append the validation Loss of this epoch in the list with L2 regularization term
                validationLoss.append((FWNN.calculateValidationLoss()+FWNN.L2regularization())/FWNN.x_valid.shape[0]);
                #calculates the training accuracy after the epoch
                accuracytrain=FWNN.calculateAccuracy("train")
                trainingaccuracy.append(accuracytrain)
                #calculates the validation accuracy after the epoch
                accuracyvalid=FWNN.calculateAccuracy("validation")
                validationaccuracy.append(accuracyvalid)
                #wandb logs
                wandb.log({'trainingloss':trainingLoss[epoch],'validationloss':validationLoss[epoch], 'trainingaccuracy':accuracytrain, 'validationaccuracy':accuracyvalid,'epoch':epoch })
                print("At epoch:{} trainingloss: {} Training accuracy: {} validation accuracy: {}".format(epoch,trainingLoss[epoch],accuracytrain,accuracyvalid))
                
            #returns the training and test accuracy of the model after the model is trained on all epochs           
            accuracytrain=FWNN.calculateAccuracy("train")
            accuracytest=FWNN.calculateAccuracy("test")
            return trainingLoss,accuracytrain,accuracytest
        if FWNN.optimizer=="rmsprop":
            beta=FWNN.beta
            L=FWNN.noOfHL+1
            k=10
            x_train=FWNN.x_train
            y_train=FWNN.y_train
            eta=FWNN.learningRate
            batchSize=FWNN.batchSize
            v_w=[0]*(L+1)
            v_b=[0]*(L+1)
            deltaw=[]
            deltab=[]
            prev_w=[0]*(L+1)
            prev_b=[0]*(L+1)
            loss=[]
            trainingLoss=[]
            trainingaccuracy=[]
            validationaccuracy=[]
            validationLoss=[]
            t=0
            for epoch in range(FWNN.epochs):
                loss=[]
                for i in range(x_train.shape[0]):
                    if(i%batchSize==0):
                        if(i!=0):
                            if t==0:
                                #update w gradient with L2 regularization term
                                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                                v_w[1:]=[(1-beta) * deltaw[i]**2 for i in range(1, L+1)]
                                v_b[1:]=[(1-beta) * deltab[i]**2 for i in range(1, L+1)]
                                t=1
                            else:
                                #update w gradient with L2 regularization term
                                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                                v_w[1:]=[beta* prev_w[i] + (1-beta) * deltaw[i]**2 for i in range(1, L+1)]
                                v_b[1:]=[beta* prev_b[i] + (1-beta) * deltab[i]**2 for i in range(1, L+1)]
                            #update the weights and biases   
                            FWNN.W[1:] = [FWNN.W[i] - (eta * deltaw[i]) / batchSize*(v_w[i]+FWNN.epsilon)**0.5 for i in range(1, L+1)]
                            FWNN.b[1:] = [FWNN.b[i] - (eta * deltab[i]) / batchSize*(v_b[i]+FWNN.epsilon)**0.5 for i in range(1, L+1)]
                            prev_w=v_w
                            prev_b=v_b
                                    
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=w_g
                        deltab=b_g
                        
                    else:
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=FWNN.acc_grad(deltaw,w_g)
                        deltab=FWNN.acc_grad(deltab,b_g)
                        
                    #append loss for this datapoint
                    loss.append(FWNN.calculateLoss(yhat,i))
                #update w gradient with L2 regularization term
                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                #update the weights and biases for the last batch
                FWNN.W[1:] = [FWNN.W[i] - (eta * deltaw[i]) / batchSize*(v_w[i]+FWNN.epsilon)**0.5 for i in range(1, L+1)]
                FWNN.b[1:] = [FWNN.b[i] - (eta * deltab[i]) / batchSize*(v_b[i]+FWNN.epsilon)**0.5 for i in range(1, L+1)]
                #append the training Loss of this epoch in the list with L2 regularization term
                trainingLoss.append((np.sum(loss)+FWNN.L2regularization())/FWNN.x_train.shape[0])
                #append the validation Loss of this epoch in the list with L2 regularization term
                validationLoss.append((FWNN.calculateValidationLoss()+FWNN.L2regularization())/FWNN.x_valid.shape[0]);
                #calculates the training accuracy after the epoch
                accuracytrain=FWNN.calculateAccuracy("train")
                trainingaccuracy.append(accuracytrain)
                #calculates the validation accuracy after the epoch
                accuracyvalid=FWNN.calculateAccuracy("validation")
                validationaccuracy.append(accuracyvalid)
                #wandb logs
                wandb.log({'trainingloss':trainingLoss[epoch],'validationloss':validationLoss[epoch], 'trainingaccuracy':accuracytrain, 'validationaccuracy':accuracyvalid,'epoch':epoch })
                print("At epoch:{} trainingloss: {} Training accuracy: {} validation accuracy: {}".format(epoch,trainingLoss[epoch],accuracytrain,accuracyvalid))
                
            #returns the training and test accuracy of the model after the model is trained on all epochs         
            accuracytrain=FWNN.calculateAccuracy("train")
            accuracytest=FWNN.calculateAccuracy("test")
            return trainingLoss,accuracytrain,accuracytest
        if FWNN.optimizer=="adam":
            Beta1=FWNN.Beta1;
            Beta2=FWNN.Beta2;
            epochs=FWNN.epochs
            L=FWNN.noOfHL+1
            k=10
            x_train=FWNN.x_train
            y_train=FWNN.y_train
            eta=FWNN.learningRate
            batchSize=FWNN.batchSize
            deltaw=[]
            deltab=[]
            v_w=[0]*(L+1)
            v_b=[0]*(L+1)
            m_w=[0]*(L+1)
            m_b=[0]*(L+1)
            v_w_hat=[0]*(L+1)
            v_b_hat=[0]*(L+1)
            m_w_hat=[0]*(L+1)
            m_b_hat=[0]*(L+1)
            loss=[]
            trainingLoss=[]
            trainingaccuracy=[]
            validationaccuracy=[]
            validationLoss=[]
            t=1
            for epoch in range(epochs):
                loss=[]
                for i in range(x_train.shape[0]):
                    if(i%batchSize==0):
                        if(i!=0):
                            #update the weights and biases
                            if t==1:
                                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                                m_w[1:] = [(1-Beta1)* deltaw[i]/batchSize for i in range(1, L+1)]
                                m_b[1:] = [(1-Beta1)* deltab[i]/batchSize for i in range(1, L+1)]
                                v_w[1:]=[(1-Beta2)*np.multiply(deltaw[i]/batchSize,deltaw[i]/batchSize) for i in range(1, L+1)]
                                v_b[1:]=[(1-Beta2)*np.multiply(deltab[i]/batchSize,deltab[i]/batchSize) for i in range(1, L+1)]
                                
                                m_w_hat[1:]=[m_w[i]/(1-np.power(Beta1,t)) for i in range(1, L+1)]
                                m_b_hat[1:]=[m_b[i]/(1-np.power(Beta1,t)) for i in range(1, L+1)]
                                v_w_hat[1:]=[v_w[i]/(1-np.power(Beta2,t)) for i in range(1, L+1)]
                                v_b_hat[1:]=[v_b[i]/(1-np.power(Beta2,t)) for i in range(1, L+1)]
                                
                                FWNN.W[1:] = [FWNN.W[i] -  eta * m_w_hat[i]/(np.sqrt(v_w_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                                FWNN.b[1:] = [FWNN.b[i] -  eta * m_b_hat[i]/(np.sqrt(v_b_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                                t+=1
                            else:
                                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                                m_w[1:] = [Beta1*m_w[i]+(1-Beta1)* deltaw[i]/batchSize for i in range(1, L+1)]
                                m_b[1:] = [Beta1*m_b[i]+(1-Beta1)* deltab[i]/batchSize for i in range(1, L+1)]
                                v_w[1:] = [Beta2*v_w[i]+(1-Beta2)*np.multiply(deltaw[i]/batchSize,deltaw[i]/batchSize) for i in range(1, L+1)]
                                v_b[1:] = [Beta2*v_b[i]+(1-Beta2)*np.multiply(deltab[i]/batchSize,deltab[i]/batchSize) for i in range(1, L+1)]
                                
                                m_w_hat[1:]=[m_w[i]/(1-np.power(Beta1,t)) for i in range(1, L+1)]
                                m_b_hat[1:]=[m_b[i]/(1-np.power(Beta1,t)) for i in range(1, L+1)] 
                                v_w_hat[1:]=[v_w[i]/(1-np.power(Beta2,t)) for i in range(1, L+1)]
                                v_b_hat[1:]=[v_b[i]/(1-np.power(Beta2,t)) for i in range(1, L+1)]
                                
                                FWNN.W[1:] = [FWNN.W[i] -  eta * m_w_hat[i]/(np.sqrt(v_w_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                                FWNN.b[1:] = [FWNN.b[i] -  eta * m_b_hat[i]/(np.sqrt(v_b_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                                t+=1
                                
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=w_g
                        deltab=b_g
                    else:
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=FWNN.acc_grad(deltaw,w_g)
                        deltab=FWNN.acc_grad(deltab,b_g)
                        
                    #append loss for this datapoint
                    loss.append(FWNN.calculateLoss(yhat,i))
                #update the weights and biases for the last batch
                FWNN.W[1:] = [FWNN.W[i] -  eta * m_w_hat[i]/(np.sqrt(v_w_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                FWNN.b[1:] = [FWNN.b[i] -  eta * m_b_hat[i]/(np.sqrt(v_b_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                #append the training Loss of this epoch in the list with L2 regularization term
                trainingLoss.append((np.sum(loss)+FWNN.L2regularization())/FWNN.x_train.shape[0])
                #append the validation Loss of this epoch in the list with L2 regularization term
                validationLoss.append((FWNN.calculateValidationLoss()+FWNN.L2regularization())/FWNN.x_valid.shape[0]);
                #calculates the training accuracy after the epoch
                accuracytrain=FWNN.calculateAccuracy("train")
                trainingaccuracy.append(accuracytrain)
                #calculates the validation accuracy after the epoch
                accuracyvalid=FWNN.calculateAccuracy("validation")
                validationaccuracy.append(accuracyvalid)
                #wandb logs
                wandb.log({'trainingloss':trainingLoss[epoch],'validationloss':validationLoss[epoch], 'trainingaccuracy':accuracytrain, 'validationaccuracy':accuracyvalid,'epoch':epoch })
                print("At epoch:{} trainingloss: {} Training accuracy: {} validation accuracy: {}".format(epoch,trainingLoss[epoch],accuracytrain,accuracyvalid))
            #returns the training and test accuracy of the model after the model is trained on all epochs    
            accuracytrain=FWNN.calculateAccuracy("train")
            accuracytest=FWNN.calculateAccuracy("test")
            return trainingLoss,accuracytrain,accuracytest
        if FWNN.optimizer=="nadam":
            Beta1=FWNN.Beta1;
            Beta2=FWNN.Beta2;
            epochs=FWNN.epochs
            L=FWNN.noOfHL+1
            k=10
            x_train=FWNN.x_train
            y_train=FWNN.y_train
            eta=FWNN.learningRate
            batchSize=FWNN.batchSize
            deltaw=[]
            deltab=[]
            v_w=[0]*(L+1)
            v_b=[0]*(L+1)
            m_w=[0]*(L+1)
            m_b=[0]*(L+1)
            v_w_hat=[0]*(L+1)
            v_b_hat=[0]*(L+1)
            m_w_hat=[0]*(L+1)
            m_b_hat=[0]*(L+1)
            W_l=FWNN.W
            b_l=FWNN.b
            loss=[]
            trainingLoss=[]
            trainingaccuracy=[]
            validationaccuracy=[]
            validationLoss=[]
            t=0
            for epoch in range(epochs):
                loss=[]
                for i in range(x_train.shape[0]):
                    if i!=0:
                        #reassigning W_l and b_l to FWNN.W and FWNN.b for look aheads
                        FWNN.W=temp_w
                        FWNN.b=temp_b
                    if(i%batchSize==0):
                        if(i!=0):
                            #update the weights and biases
                            if t==0:
                                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                                m_w[1:] = [(1-Beta1)* deltaw[i]/batchSize for i in range(1, L+1)]
                                m_b[1:] = [(1-Beta1)* deltab[i]/batchSize for i in range(1, L+1)]
                                v_w[1:]=[(1-Beta2)*np.multiply(deltaw[i]/batchSize,deltaw[i]/batchSize) for i in range(1, L+1)]
                                v_b[1:]=[(1-Beta2)*np.multiply(deltab[i]/batchSize,deltab[i]/batchSize) for i in range(1, L+1)]
                                
                                m_w_hat[1:]=[m_w[i]/(1-np.power(Beta1,1)) for i in range(1, L+1)]
                                m_b_hat[1:]=[m_b[i]/(1-np.power(Beta1,1)) for i in range(1, L+1)]
                                v_w_hat[1:]=[v_w[i]/(1-np.power(Beta2,1)) for i in range(1, L+1)]
                                v_b_hat[1:]=[v_b[i]/(1-np.power(Beta2,1)) for i in range(1, L+1)]
                                
                                FWNN.W[1:] = [FWNN.W[i] -  eta * m_w_hat[i]/(np.sqrt(v_w_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                                FWNN.b[1:] = [FWNN.b[i] -  eta * m_b_hat[i]/(np.sqrt(v_b_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                                t+=1
                            else:
                                deltaw[1:]=[deltaw[i]+FWNN.weightDecay*FWNN.W[i] for i in range(1,L+1)]
                                m_w[1:] = [Beta1*m_w[i]+(1-Beta1)* deltaw[i]/batchSize for i in range(1, L+1)]
                                m_b[1:] = [Beta1*m_b[i]+(1-Beta1)* deltab[i]/batchSize for i in range(1, L+1)]
                                v_w[1:] = [Beta2*v_w[i]+(1-Beta2)*np.multiply(deltaw[i]/batchSize,deltaw[i]/batchSize) for i in range(1, L+1)]
                                v_b[1:] = [Beta2*v_b[i]+(1-Beta2)*np.multiply(deltab[i]/batchSize,deltab[i]/batchSize) for i in range(1, L+1)]
                                
                                m_w_hat[1:]=[m_w[i]/(1-np.power(Beta1,t)) for i in range(1, L+1)]
                                m_b_hat[1:]=[m_b[i]/(1-np.power(Beta1,t)) for i in range(1, L+1)] 
                                v_w_hat[1:]=[v_w[i]/(1-np.power(Beta2,t)) for i in range(1, L+1)]
                                v_b_hat[1:]=[v_b[i]/(1-np.power(Beta2,t)) for i in range(1, L+1)]
                                
                                W_l[1:] = [W_l[i] -  eta * m_w_hat[i]/(np.sqrt(v_w_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                                b_l[1:] = [b_l[i] -  eta * m_b_hat[i]/(np.sqrt(v_b_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                                #update the lookaheads
                                FWNN.W[1:] = [FWNN.W[i] -  eta * m_w_hat[i]/(np.sqrt(v_w_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                                FWNN.b[1:] = [FWNN.b[i] -  eta * m_b_hat[i]/(np.sqrt(v_b_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                                t+=1
                                
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=w_g
                        deltab=b_g
                    else:
                        Hs,As,yhat=FWNN.forwardPropogation(x_train[i])
                        w_g,b_g=FWNN.backwardPropogation(i,Hs,As,yhat,y_train)
                        deltaw=FWNN.acc_grad(deltaw,w_g)
                        deltab=FWNN.acc_grad(deltab,b_g)
                    #storing W_l and b_l in temp and assigning it to FWNN.W and FWNN.b to calculate the loss
                    temp_w=W_l
                    temp_b=b_l
                    FWNN.W=W_l
                    FWNN.b=b_l
                        
                    #append loss for this datapoint
                    loss.append(FWNN.calculateLoss(yhat,i))
                #update the weights and biases for the last batch
                FWNN.W[1:] = [FWNN.W[i] -  eta * m_w_hat[i]/(np.sqrt(v_w_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                FWNN.b[1:] = [FWNN.b[i] -  eta * m_b_hat[i]/(np.sqrt(v_b_hat[i]+FWNN.epsilon)) for i in range(1, L+1)]
                #append the training Loss of this epoch in the list with L2 regularization term
                trainingLoss.append((np.sum(loss)+FWNN.L2regularization())/FWNN.x_train.shape[0])
                #append the validation Loss of this epoch in the list with L2 regularization term
                validationLoss.append((FWNN.calculateValidationLoss()+FWNN.L2regularization())/FWNN.x_valid.shape[0])
                #calculates the training accuracy after the epoch
                accuracytrain=FWNN.calculateAccuracy("train")
                trainingaccuracy.append(accuracytrain)
                #calculates the validation accuracy after the epoch
                accuracyvalid=FWNN.calculateAccuracy("validation")
                validationaccuracy.append(accuracyvalid)
                #wandb logs
                wandb.log({'trainingloss':trainingLoss[epoch],'validationloss':validationLoss[epoch], 'trainingaccuracy':accuracytrain, 'validationaccuracy':accuracyvalid,'epoch':epoch })
                print("At epoch:{} trainingloss: {} Training accuracy: {} validation accuracy: {}".format(epoch,trainingLoss[epoch],accuracytrain,accuracyvalid))
            #returns the training and test accuracy of the model after the model is trained on all epochs   
            accuracytrain=FWNN.calculateAccuracy("train")
            accuracytest=FWNN.calculateAccuracy("test")
            return trainingLoss,accuracytrain,accuracytest