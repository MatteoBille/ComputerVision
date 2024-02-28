clear all;
tic;
load('Data/SVM/Learned_Data_SVM_KNN.mat')


Places=Learned_Data_SVM(1).Total_Class';

dirTest = dir(fullfile('Dataset/test/*','*.jpg')); 
num_images=numel(dirTest);
%% %read image and extract surf point with SURF detector and feature with SURF descriptor 

for i= 1:num_images
%         RS(i).name
    out=regexp(dirTest(i).folder,'\','split');

    TestImageData(i).Class= out(:,end);
    I=imread(fullfile(dirTest(i).folder,dirTest(i).name));
    I=imresize(I,[256,256]);
    TestImageData(i).I = I;
    TestImageData(i).points = detectSURFFeatures(TestImageData(i).I,'NumScaleLevels',4,'MetricThreshold',100);
    [TestImageData(i).features, TestImageData(i).valid_points] = extractFeatures(TestImageData(i).I, TestImageData(i).points,'FeatureSize',128);        

end

%% %Create histograms for each image


for cc=1:length(Learned_Data_SVM)
    num_clusters=Learned_Data_SVM(cc).Num_cluster;
    t=mat2cell(zeros(num_clusters,num_images)',ones(num_images,1));
    [TestImageData(1:num_images).dictionary]=t{:};
    [TestImageData(1:num_images).point_cluster]=t{:};

    Histograms_Points=[];
    All_plausibleTest=[];


        [TestImageData(1:num_images).dictionary]= deal([]);
        [TestImageData(1:num_images).point_cluster]=deal([]);


        Ksigma= @(sigma,x)1/sqrt(2*pi)*sigma*exp(-x./sigma^2);

    for i=1:num_images

        k=1;
        [index,d]=knnsearch(Learned_Data_SVM(cc).Centroid,TestImageData(i).features,'K',k);
        index=index';
        index=index(:);
        TestImageData(i).point_cluster=[index];

        TestImageData(i).dictionary= accumarray(index,1,[num_clusters 1])'/length(TestImageData(i).features);
        [Histograms_Points]=[Histograms_Points;TestImageData(i).dictionary];   


        s=1;
        values=Ksigma(s,d);
        plausiblTest=zeros(1,num_clusters);
        for jj=1:length(index)
            plausiblTest(index(jj))=plausiblTest(index(jj))+values(jj);
        end

        plausiblTest=plausiblTest./length(TestImageData(i).features);
        [All_plausibleTest]=[All_plausibleTest;plausiblTest];  
    end

    Histograms_Points=Histograms_Points.*(Learned_Data_SVM(cc).Total_Histogram~=0).*Learned_Data_SVM(cc).idf_array;
    All_plausibleTest=All_plausibleTest.*(Learned_Data_SVM(cc).Total_Histogram~=0).*Learned_Data_SVM(cc).idf_array;
    
    Histograms_Points=Histograms_Points./vecnorm(Histograms_Points')';
    All_plausibleTest=All_plausibleTest./vecnorm(All_plausibleTest')';

%% %Evaluation of each svm and create confusion matrix

    Valutazione_SVM(cc).Num_cluster=Learned_Data_SVM(cc).Num_cluster;
    
    %% %linear SVM
    Total_scores=[];

    for i = 1:length(Learned_Data_SVM(cc).SVM_Classes_linear)
        [~,score]=predict(Learned_Data_SVM(cc).SVM_Classes_linear(i).Model,Histograms_Points);
        Total_scores(:,i)=score(:,2);
    end

    [~,predicted]= max(Total_scores,[],2);
    t=num2cell(predicted);
    [TestImageData(1:length(t)).Predicted_Class]=t{:};

    Total=zeros(length(Places));
    
    for i=1:length(TestImageData)
        y=cast(find(strcmp(Places, cell2mat(TestImageData(i).Class))),'uint8');
        x=TestImageData(i).Predicted_Class;
        Total(y,x)= Total(y,x)+1;
    end
    
    Valutazione_SVM(cc).Linear.Total=Total;
    Valutazione_SVM(cc).Linear.Result=trace(Total)/sum(sum(Total));
    %% %gaussian SVM
    
     Total_scores=[];

     for i = 1:length(Learned_Data_SVM(cc).SVM_Classes_Gaussian)
         [~,score]=predict(Learned_Data_SVM(cc).SVM_Classes_Gaussian(i).Model,Histograms_Points);
         Total_scores(:,i)=score(:,2);
     end

            [~,predicted]= max(Total_scores,[],2);
            t=num2cell(predicted);
            [TestImageData(1:length(t)).Predicted_Class]=t{:};

         Total=zeros(length(Places));
         for i=1:length(TestImageData)
            y=cast(find(strcmp(Places, cell2mat(TestImageData(i).Class))),'uint8');
            x=TestImageData(i).Predicted_Class;
            Total(y,x)= Total(y,x)+1;
         end
         Valutazione_SVM(cc).Gaussian.Total=Total;
         Valutazione_SVM(cc).Gaussian.Result=trace(Total)/sum(sum(Total));
    %% %linear SVM plausible
    Total_scores=[];

    for i = 1:length(Learned_Data_SVM(cc).SVM_Classes_linear)
        [~,score]=predict(Learned_Data_SVM(cc).SVM_Classes_linear_plausible(i).Model,All_plausibleTest);
        Total_scores(:,i)=score(:,2);
    end

    [~,predicted]= max(Total_scores,[],2);
    t=num2cell(predicted);
    [TestImageData(1:length(t)).Predicted_Class]=t{:};

    Total=zeros(length(Places));
    
    for i=1:length(TestImageData)
        y=cast(find(strcmp(Places, cell2mat(TestImageData(i).Class))),'uint8');
        x=TestImageData(i).Predicted_Class;
        Total(y,x)= Total(y,x)+1;
    end
    
    Valutazione_SVM(cc).Linear_plausible.Total=Total;
    Valutazione_SVM(cc).Linear_plausible.Result=trace(Total)/sum(sum(Total));
    %% %gaussian SVM plausible
    
     Total_scores=[];

     for i = 1:length(Learned_Data_SVM(cc).SVM_Classes_Gaussian_plausible)
         [~,score]=predict(Learned_Data_SVM(cc).SVM_Classes_Gaussian_plausible(i).Model,All_plausibleTest);
         Total_scores(:,i)=score(:,2);
     end

            [~,predicted]= max(Total_scores,[],2);
            t=num2cell(predicted);
            [TestImageData(1:length(t)).Predicted_Class]=t{:};

         Total=zeros(length(Places));
         for i=1:length(TestImageData)
            y=cast(find(strcmp(Places, cell2mat(TestImageData(i).Class))),'uint8');
            x=TestImageData(i).Predicted_Class;
            Total(y,x)= Total(y,x)+1;
         end
         Valutazione_SVM(cc).Gaussian_plausible.Total=Total;
         Valutazione_SVM(cc).Gaussian_plausible.Result=trace(Total)/sum(sum(Total));
    save('Data/SVM/Valutazione_SVM','Valutazione_SVM');
end
toc;