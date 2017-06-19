function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


y_tmp = [1 2 3 4 5 6 7 8 9 10];

a1 = [ones(m, 1) X];  
z2 = Theta1 * a1';  
a2 = sigmoid(z2);  
a2 = [ones(1, m);a2];  
z3 = Theta2 * a2;  
a3 = sigmoid(z3);
a3 = a3';

[t o] = size(a3);
delta3 = zeros(t,o);
delta2 = zeros(m, hidden_layer_size);

%fprintf("a3 size %d %d\n", t, o);
%training numbers
for i=1:m
  y_i = y_tmp==y(i);
  %fprintf("%d\n", y(i));
  % labels numbers 
  for c = 1:num_labels     
     J = J - y_i(c)*log(a3(i,c)) - (1 - y_i(c))*log(1-a3(i,c));
     delta3(i,c) = a3(i,c) - y_i(c);
  end
  
end
delta2 = Theta2(:,2:end)'*delta3'.*sigmoidGradient(z2);
temp2 = Theta2;
temp2(:,1) = 0;  

temp1 = Theta1;
temp1(:,1) = 0; 

Theta2_grad = delta3' * a2'/m + temp2*lambda/m;
Theta1_grad = delta2  * a1/m + temp1*lambda/m;

grad = [Theta1_grad(:) ; Theta2_grad(:)];

r = 0;
[q1_r,q1_c] = size(Theta1);
[q2_r,q2_c] = size(Theta2);

%fprintf("Theta1 size %d %d\n", q1_r, q1_c);
%fprintf("Theta2 size %d %d\n", q2_r, q2_c);

for j=1:q1_r
   for k=2:q1_c
      r = r + Theta1(j,k)^2; 
   end
end

for j=1:q2_r
   for k=2:q2_c
      r = r + Theta2(j,k)^2; 
   end
end

J = J/m + r*lambda/(2*m);



















% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
