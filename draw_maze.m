% Usage: draw_maze(transition, reward, task, Q)
% Before: States of the maze are numbered from left
%       to right, top to bottom starting with 0.
%   transition: A stateXaction matrix where taking
%               action A in state S will result in
%               the new state transition(S,A). If
%               transition(S,A) is -1 then action A
%               is non-viable in state S.
%               Action 1 is up/north.
%               Action 2 is right/east.
%               Action 3 is down/south.
%               Action 4 is left/west.
%   reward:     A stateXtask matrix where goal states
%               of the task have a nonzero value.
%   task:       The number of tasks in the problem.
%   Q:          A (state*action)Xtask matrix where
%               the first four values in the first
%               column represent the q-values for
%               the first task and state when taking
%               the action up/north, right/east,
%               down/south or left/west respectivily
%               and so on.
% After: Only the side effect of displaying the maze
%       on a per task basis as one image.
function draw_maze(transition, reward, task, Q)
    state = length(transition);
    transition = transition + 1;
    half = ceil(task./2);
    q = zeros(state, 1);
    d = zeros(state, 1);
    a = 1;
    while transition(a, 3) == 0
        a = a + 1;
    end
    w = transition(a, 3) - a;
    h = state/w;
    walls = zeros(h*5, w*5);
    
    for i = 0:w-1
        for j = 0:h-1
            if transition(j*w + i + 1, 1) == 0
                walls(j*5 + 1, i*5 + 1: (i + 1)*5) = 1;
            end
            if transition(j*w + i + 1, 2) == 0
                walls(j*5 + 1: (j + 1)*5,(i + 1)*5) = 1;
            end
            if transition(j*w + i + 1, 3) == 0
                walls((j + 1)*5, i*5 + 1: (i + 1)*5) = 1;
            end
            if transition(j*w + i + 1, 4) == 0
                walls(j*5 + 1: (j + 1)*5, i*5 + 1) = 1;
            end
        end
    end
    
    figure(1); clf;
    for t = 1:task
        maze = zeros(h*5, w*5);
        for i = 0: state-1
%             c = 4;
%             for j = 0:3
%                if Q(i*4 + j + 1, t) == 0
%                    c = c - 1;
%                end
%             end
%             if c == 0
%                 q(i + 1) = 0;
%             else
%                 q(i + 1) = sum(Q(i*4 + 1: i*4 + 4, t))/c;
%             end
            [val,ind] = max(Q(i*4 + 1: i*4 + 4, t));
            q(i + 1) = val;
            d(i + 1) = ind;
        end
        %q = mean(reshape(Q(:,t), 4, state))
        m = max(q)*1.33;
        for i = 0:w-1
            for j = 0:h-1
                if reward(j*w + i + 1,t) == 0
                    maze(j*5 + 1: (j + 1)*5, i*5 + 1: (i + 1)*5) = q(j*w + i + 1)/m;
                    if d(j*w + i + 1) == 1
                        maze(j*5 + 1, i*5 + 3) = 0;
                    end
                    if d(j*w + i + 1) == 2
                        maze(j*5 + 3, (i + 1)*5) = 0;
                    end
                    if d(j*w + i + 1) == 3
                        maze((j + 1)*5, i*5 + 3) = 0;
                    end
                    if d(j*w + i + 1) == 4
                        maze(j*5 + 3, i*5 + 1) = 0;
                    end
                else
                    maze(j*5 + 1: (j + 1)*5, i*5 + 1: (i + 1)*5) = ...
                        [ 0 1 0 1 0
                          1 0 1 0 1
                          0 1 0 1 0
                          1 0 1 0 1
                          0 1 0 1 0 ];
                end
            end
        end
        maze = max(maze - walls, 0);
        disp(sprintf('%d: %d repeated elements, min is %5.5f and max is %5.5f', ...
            t, length(unique(maze)), min(min(maze)), max(max(maze))))
        subplot(2, half, t);
        imagesc(maze);
        t_s = 'task ';
        t_s(6) = 48 + t;
        title(t_s);
        axis image;
    end
    %colormap hot;
    shg
end