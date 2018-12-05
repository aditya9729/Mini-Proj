#Reading training data,head=first few rows,tail=last few rows,str=summary structure,dim=dimensions
X_training<-read.table(file='X_train.txt',header=FALSE,sep="")
head(X_training)
dim(X_training)
str(X_training)
names(X_training)
#Reading test data
X_test<-read.table(file='X_test.txt',header=FALSE,sep="")
head(X_test)
dim(X_test)
str(X_test)
names(X_test)
#Binding the two tables that is X_training is stacked over X_test
X<-rbind(X_training,X_test)
dim(X)
#reading labelled training data
Y_training<-read.table(file='y_train.txt',header=FALSE,sep="")
dim(Y_training)
#reading labelled test data
Y_test<-read.table(file='y_test.txt',header=FALSE,sep="")
dim(Y_test)
#binding the two dataframes,Y_training stacked over Y_test
Y<-rbind(Y_training,Y_test)
dim(Y)
names(Y)
str(Y)
#Reading subject data and subject labels
subject_training<-read.table(file='subject_train.txt',header=FALSE,sep="")
subject_test<-read.table(file='subject_test.txt',header=FALSE,sep="")
dim(subject_training)
dim(subject_test)
#Binding subject data
subject<-rbind(subject_training,subject_test)
#Changing individual column names ,completely arbitrary and sifting the columns
colnames(Y)[1]<-'activity.names'
colnames(subject)[1]<-'volunteer.under.study'
colnames(X)[1]<-'time.Body.Acc.X.mean'
colnames(X)[2]<-'time.Body.Acc.Y.mean'
colnames(X)[3]<-'time.Body.Acc.Z.mean'
colnames(X)[4]<-'time.Body.Acc.X.std'
colnames(X)[5]<-'time.Body.Acc.Y.std'
colnames(X)[6]<-'time.Body.Acc.Z.std'
colnames(X)[41]<-'time.Gravity.Acc.X.mean'
colnames(X)[42]<-'time.Gravity.Acc.Y.mean'
colnames(X)[43]<-'time.Gravity.Acc.Z.mean'
colnames(X)[44]<-'time.Gravity.Acc.X.std'
colnames(X)[45]<-'time.Gravity.Acc.Y.std'
colnames(X)[46]<-'time.Gravity.Acc.Z.std'
colnames(X)[81]<-'time.Body.Acc.Jerk.X.mean'
colnames(X)[82]<-'time.Body.Acc.Jerk.Y.mean'
colnames(X)[83]<-'time.Body.Acc.Jerk.Z.mean'
colnames(X)[84]<-'time.Body.Acc.Jerk.X.std'
colnames(X)[85]<-'time.Body.Acc.Jerk.Y.std'
colnames(X)[86]<-'time.Body.Acc.Jerk.Z.std'
colnames(X)[121]<-'time.Body.Gyro.X.mean'
colnames(X)[122]<-'time.Body.Gyro.Y.mean'
colnames(X)[123]<-'time.Body.Gyro.Z.mean'
colnames(X)[124]<-'time.Body.Gyro.X.std'
colnames(X)[125]<-'time.Body.Gyro.Y.std'
colnames(X)[126]<-'time.Body.Gyro.Z.std'
colnames(X)[161]<-'time.Body.Gyro.Jerk.X.mean'
colnames(X)[162]<-'time.Body.Gyro.Jerk.Y.mean'
colnames(X)[163]<-'time.Body.Gyro.Jerk.Z.mean'
colnames(X)[164]<-'time.Body.Gyro.Jerk.X.std'
colnames(X)[165]<-'time.Body.Gyro.Jerk.Y.std'
colnames(X)[166]<-'time.Body.Gyro.Jerk.Z.std'
colnames(X)[201]<-'time.Body.Acc.Mag.mean'
colnames(X)[202]<-'time.Body.Acc.Mag.std'
colnames(X)[214]<-'time.Gravity.Acc.Mag.mean'
colnames(X)[215]<-'time.Gravity.Acc.Mag.std'
colnames(X)[227]<-'time.Body.Acc.Jerk.Mag.mean'
colnames(X)[228]<-'time.Body.Acc.Jerk.Mag.std'
colnames(X)[240]<-'time.Body.Gyro.Mag.mean'
colnames(X)[241]<-'time.Body.Gyro.Mag.std'
colnames(X)[253]<-'time.Body.Gyro.Jerk.Mag.mean'
colnames(X)[254]<-'time.Body.Gyro.Jerk.Mag.std'
colnames(X)[266]<-'freq.Body.Acc.X.mean'
colnames(X)[267]<-'freq.Body.Acc.Y.mean'
colnames(X)[268]<-'freq.Body.Acc.Z.mean'
colnames(X)[269]<-'freq.Body.Acc.X.std'
colnames(X)[270]<-'freq.Body.Acc.Y.std'
colnames(X)[271]<-'freq.Body.Acc.Z.std'
colnames(X)[345]<-'freq.Body.Acc.Jerk.X.mean'
colnames(X)[346]<-'freq.Body.Acc.Jerk.Y.mean'
colnames(X)[347]<-'freq.Body.Acc.Jerk.Z.mean'
colnames(X)[348]<-'freq.Body.Acc.Jerk.X.std'
colnames(X)[349]<-'freq.Body.Acc.Jerk.Y.std'
colnames(X)[350]<-'freq.Body.Acc.Jerk.Z.std'
colnames(X)[424]<-'freq.Body.Gyro.X.mean'
colnames(X)[425]<-'freq.Body.Gyro.Y.mean'
colnames(X)[426]<-'freq.Body.Gyro.Z.mean'
colnames(X)[427]<-'freq.Body.Gyro.X.std'
colnames(X)[428]<-'freq.Body.Gyro.Y.std'
colnames(X)[429]<-'freq.Body.Gyro.Z.std'
colnames(X)[503]<-'freq.Body.Acc.Mag.mean'
colnames(X)[504]<-'freq.Body.Acc.Mag.std'
colnames(X)[516]<-'freq.Body.Acc.Jerk.Mag.mean'
colnames(X)[517]<-'freq.Body.Acc.Jerk.Mag.std'
colnames(X)[529]<-'freq.Body.Gyro.Mag.mean'
colnames(X)[530]<-'freq.Body.Gyro.Mag.std'
colnames(X)[542]<-'freq.Body.Gyro.Jerk.Mag.mean'
colnames(X)[543]<-'freq.Body.Gyro.Jerk.Mag.std'
names(X)
#final dataset X
X_final<-X[,c(1:6,41:46,81:86,121:126,161:166,201,202,214,215,227,228,240,241,253,254,266:271,345:350,424:429,503,504,516,517,529,530,542,543)]
names(X_final)
dim(X_final)
str(X_final)
#Column binding next to each other 
X_Y<-cbind(Y,X_final)
fitset<-cbind(subject,X_Y)
#grouping by the first column and arranging in order
health<-group_by(fitset,volunteer.under.study)
health2<-arrange(fitset,fitset$volunteer.under.study,fitset$activity.names)
#Using piping we use functions such as group_by and summarize(using the function mean)to get the dataset
healthy1<-health2 %>%
  group_by(volunteer.under.study,activity.names) %>%
  summarize_all(mean)
#Converting the tibble dataset into dataframe
fit_set<-as.data.frame(healthy1)
#Converting activity names which are given as numbers into activity characters
fit_set[,2]<-sapply(fit_set[,2],function(x) {if(x==1){x='Walking'}else if(x==2){x='Walking.upstairs'}else if(x==3){x='Walking.downstairs'}else if(x==4){x='Sitting'}else if(x==5){x='Standing'}else if(x==6){x='Laying'}})
#View the final dataset
View(fit_set)