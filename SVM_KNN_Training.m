clear all;

tic
DirTrain = dir(fullfile('Dataset/train/*','*.jpg')); 
num_clusters=[15 20 50 100 250 500 750 900];
num_images=numel(DirTrain);


%% %read image and extract surf point with SURF detector and feature with SURF descriptor 
  
    for i= 1:num_images
    %     S(i).name
        out=regexp(DirTrain(i).folder,'\','split');
        
        imageData(i).Class= out(:,end);
        I = imread(fullfile(DirTrain(i).folder,DirTrain(i).name));
        I=imresize(I,[256,256]);
        imageData(i).I = I;
        imageData(i).points = detectSURFFeatures(imageData(i).I,'NumScaleLevels',4,'MetricThreshold',100);
        [imageData(i).features, imageData(i).valid_points] = extractFeatures(imageData(i).I, imageData(i).points,'FeatureSize',128);        
       
    end
    
     Total_features=cat(1,imageData(1:num_images).features);

%% %Kmeans clustering
for i =1:length(num_clusters)
    Learned_Data_SVM(i).Num_cluster=num_clusters(i);
    [~, C]= kmeans(Total_features,num_clusters(i),'MaxIter',10,'OnlinePhase','on');
    Learned_Data_SVM(i).Centroid=C;
    fprintf('Ho finito di il clustering\n');
end 

fprintf('Ho eseguito il clustering\n');
%% %Create histograms for each image
%All_Histogram is 1/N*sum(num of point in bin)
%All_plausible is 1/N*sum(Ksigma(point,distance from bin))
  
for z =1:length(Learned_Data_SVM)
    Learned_Data_SVM(z).Image_Class=cat(1,imageData.Class);
    Learned_Data_SVM(z).Total_Class=unique(Learned_Data_SVM(z).Image_Class);
    
    C=Learned_Data_SVM(z).Centroid;
    
    
    All_Histograms=[];
    All_plausible=[];
    tf_idf=[];
    
   
        [imageData(1:num_images).dictionary]= deal([]);
        [imageData(1:num_images).point_cluster]=deal([]);
        
        
        
    Ksigma= @(sigma,x)1/sqrt(2*pi)*sigma*exp(-x./sigma^2);
    
    
    for i=1:num_images
        
        %Every feature increments k bins
        k=1;
        [index,d]=knnsearch(C,imageData(i).features,'K',k);
        index=index';
        index=index(:);        
        
        imageData(i).point_cluster=[index];

        dictionary= accumarray(index,1,[Learned_Data_SVM(z).Num_cluster 1])'/length(imageData(i).features);
        [All_Histograms]=[All_Histograms;dictionary];   
        
        
        s=1;
        values=Ksigma(s,d);
        plausible=zeros(1,Learned_Data_SVM(z).Num_cluster);

        for jj=1:length(index)
            plausible(index(jj))=plausible(index(jj))+values(jj);
        end
        
        plausible=plausible./length(imageData(i).features);
        [All_plausible]=[All_plausible;plausible]; 
        
        
        Learned_Data_SVM(z).dictionary(i,:)=dictionary;
        Learned_Data_SVM(z).plausible(i,:)=plausible;
    end
    

end

fprintf('Ho finito di fare gli istogrammi\n');
%% %Find idf value to weigth each bin

for z =1:length(Learned_Data_SVM)
    Learned_Data_SVM(z).Total_Histogram=sum(cat(1,Learned_Data_SVM(z).dictionary(1:num_images,:)~=0));
    Learned_Data_SVM(z).idf_array=zeros(Learned_Data_SVM(z).Num_cluster,1);
    Learned_Data_SVM(z).idf_array=log(sum(Learned_Data_SVM(z).Total_Histogram)./Learned_Data_SVM(z).Total_Histogram);

end

%% %SVM for each class

for z =1:length(Learned_Data_SVM)
    Search_Histograms=Learned_Data_SVM(z).dictionary.*(Learned_Data_SVM(z).Total_Histogram~=0).*Learned_Data_SVM(z).idf_array;
    Plausible=Learned_Data_SVM(z).plausible.*(Learned_Data_SVM(z).Total_Histogram~=0).*Learned_Data_SVM(z).idf_array;
%     Plausible=Learned_Data_SVM(z).plausible;
%     Search_Histograms=Learned_Data_SVM(z).dictionary;
    
    Search_Histograms=Search_Histograms./vecnorm(Search_Histograms')';
    Plausible=Plausible./vecnorm(Plausible')';
    
    Places=Learned_Data_SVM(z).Total_Class';

    %% % Linear SVM
    for i=1:length(Places)
        label=[];
        Label=cast(strcmp(cat(1,Learned_Data_SVM(z).Image_Class),Places(i)),'double');
        Label(Label(:)==0)=-1;
        Learned_Data_SVM(z).SVM_Classes_linear(i).Model=fitcsvm(Search_Histograms,Label(:,1),'Standardize',true,'KernelScal','auto','BoxConstraint',50);
    end

    %% % SVM Gaussian

    for i=1:length(Places)
        label=[];
        Label=cast(strcmp(cat(1,Learned_Data_SVM(z).Image_Class),Places(i)),'double');
        Label(Label(:)==0)=-1;
        Learned_Data_SVM(z).SVM_Classes_Gaussian(i).Model=fitcsvm(Search_Histograms,Label(:,1),'KernelFunction','gaussian','Standardize',true,'KernelScal','auto','BoxConstraint',50);
    end

    
    %% % Linear SVM plausible
    for i=1:length(Places)
        label=[];
        Label=cast(strcmp(cat(1,Learned_Data_SVM(z).Image_Class),Places(i)),'double');
        Label(Label(:)==0)=-1;
        Learned_Data_SVM(z).SVM_Classes_linear_plausible (i).Model=fitcsvm(Plausible,Label(:,1),'Standardize',true,'KernelScal','auto','BoxConstraint',50);
    end

    %% % SVM Gaussian plausible


    for i=1:length(Places)
        label=[];
        Label=cast(strcmp(cat(1,Learned_Data_SVM(z).Image_Class),Places(i)),'double');
        Label(Label(:)==0)=-1;
        Learned_Data_SVM(z).SVM_Classes_Gaussian_plausible(i).Model=fitcsvm(Plausible,Label(:,1),'KernelFunction','gaussian','Standardize',true,'KernelScal','auto','BoxConstraint',50);
    end
    save('Data/SVM/Learned_Data_SVM_KNN.MAT','Learned_Data_SVM','-v7.3');
end
toc
