%% generateExamples: generate an example of a state space system
%% example input: 'butter' for a butterworth filter
function [] = generateExamples(type, systemOrder, outputs, inputs)

if nargin < 4
	inputs = 1;
	if nargin < 3
		outputs = 1;
		if nargin < 2
			systemOrder = 1;
			if nargin < 1
				type = 'random';
			end
		end
	end
end

switch type
	case 'butter'
		[A, B, C, D] = generateButter(systemOrder);
	case 'random'
		systemOrder, outputs, inputs
		sys = rss(systemOrder, outputs, inputs);
		[A, B, C, D] = [sys.a, sys.b, sys.c, sys.d]
	otherwise
		error('No valid type specified');
end

filename = strcat(...
	type, '_',...
	int2str(systemOrder), '_',...
	int2str(outputs), '_',...
	int2str(inputs), '.mat')
save(filename, 'A', 'B', 'C', 'D')

end

%% generateButter: generates a butterworth filter
function [A, B, C, D] = generateButter(systemOrder)
	[A, B, C, D] = butter(systemOrder, 300./500.);
end