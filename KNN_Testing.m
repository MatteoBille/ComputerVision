clear all;

tic;
load('Data/SVM/Learned_Data_SVM_KNN.mat');
Learned_Data=Learned_Data_SVM;


    K_vote=[1 3 5 10 15 20 50 75];

    Places=Learned_Data_SVM(1).Total_Class';   

    dirTest = dir(fullfile('Dataset/test/*','*.jpg')); 
    num_images=numel(dirTest)
    %% %read image and extract surf point with SURF detector and feature with SURF descriptor 

        for i= 1:num_images
    %         RS(i).name
            out=regexp(dirTest(i).folder,'\','split');

            TestImageData(i).Class= out(:,end);
            TestImageData(i).I = imread(fullfile(dirTest(i).folder,dirTest(i).name));
            TestImageData(i).I = imresize(TestImageData(i).I,[256 256]);
            TestImageData(i).points = detectSURFFeatures(TestImageData(i).I,'NumScaleLevels',4,'MetricThreshold',100);
            [TestImageData(i).features, TestImageData(i).valid_points] = extractFeatures(TestImageData(i).I, TestImageData(i).points,'FeatureSize',128);        

        end
    %% % testing for different number of bins
for cc=1:length(Learned_Data)
    
    num_clusters=Learned_Data(cc).Num_cluster;
    
    t=mat2cell(zeros(num_clusters,num_images)',ones(num_images,1));
    [TestImageData(1:num_images).dictionary]=t{:};
    [TestImageData(1:num_images).point_cluster]=t{:};


%% %Create histograms for each image

    num_clusters=Learned_Data_SVM(cc).Num_cluster;
    t=mat2cell(zeros(num_clusters,num_images)',ones(num_images,1));
    
    [TestImageData(1:num_images).dictionary]=t{:};
    [TestImageData(1:num_images).point_cluster]=t{:};

    standard_histograms_search=[];
    plausible_histograms_search=[];

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
        [standard_histograms_search]=[standard_histograms_search;TestImageData(i).dictionary];   


        s=1;
        values=Ksigma(s,d);
        plausiblTest=zeros(1,num_clusters);
        for jj=1:length(index)
            plausiblTest(index(jj))=plausiblTest(index(jj))+values(jj);
        end

        plausiblTest=plausiblTest./length(TestImageData(i).features);
        [plausible_histograms_search]=[plausible_histograms_search;plausiblTest];  
    end

    standard_histograms_search=standard_histograms_search.*(Learned_Data_SVM(cc).Total_Histogram~=0).*Learned_Data_SVM(cc).idf_array;
    plausible_histograms_search=plausible_histograms_search.*(Learned_Data_SVM(cc).Total_Histogram~=0).*Learned_Data_SVM(cc).idf_array;
    
    standard_histograms_search=standard_histograms_search./vecnorm(standard_histograms_search')';
    plausible_histograms_search=plausible_histograms_search./vecnorm(plausible_histograms_search')';

     %% Evaluation of KNN

        
    Search_Histograms=Learned_Data(cc).dictionary.*Learned_Data(cc).idf_array;
    Plausible_Histograms=Learned_Data(cc).plausible.*Learned_Data(cc).idf_array;

    Search_Histograms=Search_Histograms./vecnorm(Search_Histograms')';
    Plausible_Histograms=Plausible_Histograms./vecnorm(Plausible_Histograms')';
        
    for z= 1:length(K_vote)
        %% STANDARD Histograms
        K_vote(z);
        idx=knnsearch(Search_Histograms,standard_histograms_search,'K',K_vote(z),'NSmethod','exhaustive');

        
         for i= 1:length(TestImageData)
             %concateno gli array del voto
             voting_vector(i,:) = cat(1,Learned_Data(cc).Image_Class(idx(i,:)))';
             for j=1:length(Places)
                 TestImageData(i).Vote(j)=sum(strcmp(voting_vector(i,:),Places(j)));
             end
         end


         [~,predicted]=max(cat(1,TestImageData.Vote),[],2);
         t=num2cell(predicted);
         [TestImageData(1:length(t)).Predicted_Class]=t{:};


         Total=zeros(length(Places));
         for i=1:length(TestImageData)
            y=cast(find(strcmp(Places, cell2mat(TestImageData(i).Class))),'uint8');
            x=TestImageData(i).Predicted_Class;
            Total(y,x)= Total(y,x)+1;
         end

         Result=trace(Total)/sum(sum(Total));
         Valutazione(cc).Num_Cluster=num_clusters;
         Valutazione(cc).Totale_standard(z).Kvotes=K_vote(z);
         Valutazione(cc).Totale_standard(z).Matrix=Total;
         Valutazione(cc).Totale_standard(z).Accuracy=Result;
         
          clear voting_vector;
         %% PLAUSIBLE Histograms
         
         idx=knnsearch(Plausible_Histograms,plausible_histograms_search,'K',K_vote(z),'NSmethod','exhaustive');
         
          for i= 1:length(TestImageData)
             %concateno gli array del voto
             voting_vector(i,:) = cat(1,Learned_Data(cc).Image_Class(idx(i,:)))';
             for j=1:length(Places)
                 TestImageData(i).Vote(j)=sum(strcmp(voting_vector(i,:),Places(j)));
             end
         end


         [~,predicted]=max(cat(1,TestImageData.Vote),[],2);
         t=num2cell(predicted);
         [TestImageData(1:length(t)).Predicted_Class]=t{:};


         Total=zeros(length(Places));
         for i=1:length(TestImageData)
            y=cast(find(strcmp(Places, cell2mat(TestImageData(i).Class))),'uint8');
            x=TestImageData(i).Predicted_Class;
            Total(y,x)= Total(y,x)+1;
         end
         
         Result=trace(Total)/sum(sum(Total));
         Valutazione(cc).Totale_plausible(z).Kvotes=K_vote(z);
         Valutazione(cc).Totale_plausible(z).Matrix=Total;
         Valutazione(cc).Totale_plausible(z).Accuracy=Result;
        
         clear voting_vector;
    end
    save('Data/KNN/Valutazione_KNN.mat','Valutazione');
    % % % % 
end
toc