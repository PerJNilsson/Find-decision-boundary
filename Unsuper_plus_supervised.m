clear all;
clf;
clc;

nUpdates = 10e2;
learningRate = 0.2;
beta = 1/2;
nRuns = 1;
tic
dataTask3 = load('data_ex2_task3_2017.m');
classification = dataTask3(:,1);
dataUncentered = dataTask3(:,2:3);
dimension = size(dataUncentered,2);
nInputPatterns = length(dataTask3);

minClassificationError = 1.0;
classAvg = 0;

hold on
figure(1)

for k = 1:length(dataUncentered)
  if classification(k) == 1
    plot(dataUncentered(k,1), dataUncentered(k,2), 'r.') %, 'FaceAlpha', 0.2
  end
  if classification(k) == -1
    plot(dataUncentered(k,1), dataUncentered(k,2), 'b.') %, 'FaceAlpha', 0.2
  end
end
title('Input data')
hold off

for o=4
  gaussianNeurons = [1 1 1 15];
  nNeurons = gaussianNeurons(o);
  
  for n = 1:nRuns
    weightsUnsupervised = 2*rand(nNeurons, dimension)-1;
    gSpace = zeros(nNeurons, nInputPatterns);
    
    %Feed with a random pattern
    for i = 1:nUpdates
      randIndex = randi(nInputPatterns);
      x = dataUncentered(randIndex, :);
      winningNeuron= GetWinningNeuron( weightsUnsupervised, x );
      deltaWeight = learningRate*(x-weightsUnsupervised(winningNeuron,:));
      weightsUnsupervised(winningNeuron,:) = weightsUnsupervised(winningNeuron,:) + deltaWeight;
    end
    
    
    % Computing gSpace
    for j = 1:nInputPatterns
      for k = 1:nNeurons
        gSpace(k,j) = ActivationFunction(dataUncentered(j,:), weightsUnsupervised, k );
      end
    end
    
    weightsPerceptron =  2*rand(1, nNeurons)-1;
    threshold = 2*rand-1;
    deltaH = zeros(nInputPatterns,1);
    learningRatePerceptron = 0.1;
    % Simple perception learning
    for i=1:3000
      output = tanh(beta*(weightsPerceptron*gSpace - threshold));
      deltaH =  (classification' - output);
      deltaHsquare = deltaH.^2;
      networkH = 0.5*sum(deltaHsquare);
      
      randIndex2 = randi(nInputPatterns);
      randPattern = gSpace(:,randIndex2);
      
      gPrime = beta*(1-tanh(beta*(weightsPerceptron*randPattern-threshold))^2);
      
      deltaW = learningRatePerceptron*gPrime*deltaH(randIndex2)*randPattern';
      deltaThreshold = -learningRatePerceptron*gPrime*deltaH(randIndex2);
      weightsPerceptron = weightsPerceptron + deltaW;
      threshold = threshold + deltaThreshold;
    end
    
    classificationError = (1/(2*nInputPatterns))*sum(abs(classification(:,1)'-sign(output(1,:))));
    
    classAvg(n) = classificationError;
  end
  classificationAverage(o) = mean(classAvg);
%end
%plot(gaussianNeurons(:), classificationAverage(:))
%
if minClassificationError > classificationError
  minClassificationError = classificationError;
  bestWeightsSupervised = weightsUnsupervised;
  bestWeightsPerceptron = weightsPerceptron;
  bestThreshold = threshold;
  hold on
  for k = 1:nInputPatterns
    if output(k) < 0
      plot(dataUncentered(k,1), dataUncentered(k,2), 'ro')
    end
    if output(k) > 0
      plot(dataUncentered(k,1), dataUncentered(k,2), 'bo')
    end
  end
end
end
%end
classAvgerage = mean(classAvg);
[X,Y] = meshgrid(-15:0.4:25, -10:0.25:15);

points = [X(:), Y(:)];
nPoints = size(points,1);

% Computing gSpaceGrid
for j = 1:nPoints
  for k = 1:nNeurons
    gSpaceGrid(k,j) = ActivationFunction(points(j,:), bestWeightsSupervised, k );
  end
end

outputGrid = tanh(beta*(bestWeightsPerceptron*gSpaceGrid - bestThreshold));

hold on
figure(2)
title('Decision boundary and the trained weights and black +')
for k = 1:nPoints
  if outputGrid(k) < 0
    plot(points(k,1), points(k,2), 'r.') %, 'FaceAlpha', 0.2
  end
  if outputGrid(k) > 0
    plot(points(k,1), points(k,2), 'b.') %, 'FaceAlpha', 0.2
  end
end
plot(bestWeightsSupervised(:,1), bestWeightsSupervised(:,2), 'ks', 'Linewidth', 8)

toc
