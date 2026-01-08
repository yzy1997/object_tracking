clear all;
m_min = 1;
m_max = 6;
yaxis_thd = 9; % 行列值的邻近范围
distance_threshold = 5; % 设定距离阈值
accutimes_1 = 5;
min_distance1 = 100; max_distance1 = 750;

allData = [];
for i = m_min:m_max
    filename1 = ['valid_framedata_0113_' num2str(i) '.txt'];
    fullpath = fullfile('C:\Users\GGK\Documents\MATLAB\SPD\20250113\1', filename1);
    fileID1 = fopen(fullpath);
    data = fscanf(fileID1, '%x');
    allData = [allData; data];
    fclose(fileID1);
end

decData11 = allData;
decData1 = reshape(decData11, 5, []);
decData1 = decData1';
Data_x = decData1(:,4)*16*16+decData1(:,5);
Data_a = [decData1(:,1),decData1(:,2),decData1(:,3),Data_x];
condition = Data_a(:, 4) == 0;
Data_a(condition, :) = [];
Data_a = [Data_a(:,1),Data_a(:,2),Data_a(:,3),(Data_a(:,4)+0)*0.6];
Data_b = [Data_a(:,2),Data_a(:,3),Data_a(:,4) + min_distance1];

condition = (Data_b(:, 3) == 0 | Data_b(:, 3) < 200 );
Data_b(condition, :) = [];

figure(1);
subplot(2,2,1)
pdd1 = 0;
Data_bb = Data_b;
Data_bb(:,1) = 132-Data_b(:,1);
Data_bb(:,2) = 132-Data_b(:,2);
Data_bb(:,3) = Data_b(:,3);

[~, idx] = unique(Data_bb, 'rows', 'stable');
uniqueRows = Data_bb(idx, :);
Data_bb = uniqueRows;

B = Data_bb;
w = size(B);
z1 = 1:w(1);
z2 = [z1',B];
f1 = 0;
f2 = 0;
q11 = 0; q12 = 0; q13 = 0; q14 = 0; q15 = 0;
t11 = 0; t12 = 0; t13 = 0; t14 = 0; t15 = 0;
i = 0;

for i =1:w(1)
    if (z2(i, 4) ~= 0)
        x1 = z2(i, 2);  %ow
        x2 = z2(i, 3);  %Column 
        x3 = z2(i, 4);

        h1 = x2 - yaxis_thd; 
        h2 = x2 + yaxis_thd; 

        if(h1 < 1)
            h1 = 1;
        end 
        if(h2 > w(1))
            h2 = w(1);
        end 

        for t1 = h1:h2
            index = (z2(:,3) == t1) & (z2(:,2) == x1);
            matchingRows = z2(index, :);
            f11 = matchingRows(:, 4); 
            if f11 ~= 0;
             q11 = q11+1; 
            end
        end

        for t1 = h1:h2
            index = (z2(:,3) == t1) & (z2(:,2) == x1+1);
            matchingRows = z2(index, :);
            f12 = matchingRows(:, 4); 
            if f12 ~= 0;
             q12 = q12+1; 
            end
        end

        for t1 = h1:h2
            index = (z2(:,3) == t1) & (z2(:,2) == x1-1);
            matchingRows = z2(index, :);
            f14 = matchingRows(:, 4); 
            if f14 ~= 0;
             q14 = q14+1; 
            end
        end

        if (q11 == 1 & q12 == 0 & q14 == 0 )
            z2(i, 4) = 0;
        end
        q11 = 0;
        q12 = 0;
        q13 = 0;        
        q14 = 0;
        q15 = 0;
    end
end

Data_bb = [z2(:,2),z2(:,3),z2(:,4)];
condition = Data_bb(:, 3) == 0;
Data_bb(condition, :) = [];

A=size(Data_bb);
pdd1 = 17.317-0.012*accutimes_1*accutimes_1*accutimes_1+0.394*accutimes_1*accutimes_1-4.214*accutimes_1;
pdd1 = floor(pdd1);

for i = 1:A(1);
    if (mod(Data_bb(i,1), 2) == 0)
        Data_bb(i,2) = (Data_bb(i,2) - pdd1);
    else 
        Data_bb(i,2) = (Data_bb(i,2) + pdd1);
    end
end

% 应用中值滤波
Data_bb(:,3) = medfilt1(Data_bb(:,3), 3);

% 连通区域分析
bw = Data_bb(:,3) > 0;
cc = bwconncomp(bw);
stats = regionprops(cc, 'Area');
idx = find([stats.Area] > 10); % 去除面积小于10的连通区域
bw = ismember(labelmatrix(cc), idx);
Data_bb = Data_bb(bw, :);

% 增加距离阈值判断，结合行列值和距离值综合判断
% distance_threshold = 20; % 设定距离阈值
noise_points = false(size(Data_bb, 1), 1); % 初始化噪声点标记

for i = 1:size(Data_bb, 1)
    current_point = Data_bb(i, :);
    % 查找邻近点：行列值在 yaxis_thd 范围内
    neighbors = Data_bb(abs(Data_bb(:,1) - current_point(1)) <= yaxis_thd & ...
                abs(Data_bb(:,2) - current_point(2)) <= yaxis_thd, :);
    if size(neighbors, 1) > 1
        % 计算当前点与邻近点的距离值差异
        distance_diff = abs(neighbors(:,3) - current_point(3));
        % 如果当前点与所有邻近点的距离值差异都大于阈值，则判定为噪声点
        if all(distance_diff > distance_threshold)
            noise_points(i) = true; % 标记为噪声点
        end
    end
end

Data_bb(noise_points, :) = []; % 删除噪声点

% 绘制图像
scatter(Data_bb(:,2), Data_bb(:,1), 15, Data_bb(:,3),'s', 'filled');
clim([min_distance1, max_distance1]);
colorbar;
grid on;

% 设置图形属性和标题
% title('(a)');
xlabel('x-axis');
ylabel('y-axis');
text(66,123, '(a)', 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
set(gca,'FontSize',14, 'Fontname', 'Times New Roman');
xlim([0 134]);
ylim([0 134]);

% 设置颜色映射
% mincolor    = [0 0 1]; % red
% mediancolor = [1 0 0]; % yellow   
% maxcolor    = [0 1 0]; % blue 

% mincolor    = [1 0 0]; % red
% mediancolor = [0 1 0]; % yellow   
% maxcolor    = [0 0 1]; % blue 

mincolor    = [0 0 1]; % red
mediancolor = [0 1 0]; % yellow   
maxcolor    = [1 0 0]; % blue 


ColorMapSize = 100;
int1 = zeros(ColorMapSize,3); 
int2 = zeros(ColorMapSize,3);
for k=1:3
    int1(:,k) = linspace(mincolor(k), mediancolor(k), ColorMapSize);
    int2(:,k) = linspace(mediancolor(k), maxcolor(k), ColorMapSize);
end
meep = [int1(1:end-1,:); int2];
colormap(meep);


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 绘制3D散点图
figure(1); % 创建一个新的图形窗口
subplot(2, 2, 2);
scatter3(Data_bb(:,1), Data_bb(:,2), Data_bb(:,3), 11, Data_bb(:,3), 's', 'filled');
% scatter3(Data_bb(:,1), Data_bb(:,2), Data_bb(:,3), 18, Data_bb(:,3), 's', 'filled');
% scatter3(X, Y, Z, 点大小, 颜色数据, 形状, 'filled')

% 设置颜色映射
colormap(meep); % 使用之前定义的颜色映射
clim([min_distance1, max_distance1]); % 设置colorbar的范围
colorbar; % 添加颜色刻度尺

% 设置图形属性和标题
% title('(b)');
xlabel('x-axis');
ylabel('y-axis');
zlabel('distance'); % 添加Z轴标签
text(66,123, 680,'(b)', 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
set(gca,'FontSize',14, 'Fontname', 'Times New Roman');
grid on;

% 设置坐标轴范围
xlim([0 134]);
ylim([0 134]);
zlim([min_distance1, max_distance1]); % 设置Z轴范围

% 其他部分代码保持不变...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

m = 18;
yaxis_thd = 2;
accutimes_1 = 5;
distance_threshold = 10;
min_distance1 = 100; max_distance1 = 750;

allData = [];
for i = 3:m
    filename1 = ['valid_framedata_0113_' num2str(i) '.txt'];
    fullpath = fullfile('C:\Users\GGK\Documents\MATLAB\SPD\20250113\3', filename1);
    fileID1 = fopen(fullpath);
    data = fscanf(fileID1, '%x');
    allData = [allData; data];
    fclose(fileID1);
end

decData11 = allData;
decData1 = reshape(decData11, 5, []);
decData1 = decData1';
Data_x = decData1(:,4)*16*16+decData1(:,5);
Data_a = [decData1(:,1),decData1(:,2),decData1(:,3),Data_x];
condition = Data_a(:, 4) == 0;
Data_a(condition, :) = [];
Data_a = [Data_a(:,1),Data_a(:,2),Data_a(:,3),(Data_a(:,4)+0)*0.6];
Data_b = [Data_a(:,2),Data_a(:,3),Data_a(:,4) + min_distance1];

condition = (Data_b(:, 3) == 0 | Data_b(:, 3) < 200 );
Data_b(condition, :) = [];

figure(1);
subplot(2,2,3)
pdd1 = 0;
Data_bb = Data_b;
Data_bb(:,1) = 132-Data_b(:,1);
Data_bb(:,2) = 132-Data_b(:,2);
Data_bb(:,3) = Data_b(:,3);

[~, idx] = unique(Data_bb, 'rows', 'stable');
uniqueRows = Data_bb(idx, :);
Data_bb = uniqueRows;

B = Data_bb;
w = size(B);
z1 = 1:w(1);
z2 = [z1',B];
f1 = 0;
f2 = 0;
q11 = 0; q12 = 0; q13 = 0; q14 = 0; q15 = 0;
t11 = 0; t12 = 0; t13 = 0; t14 = 0; t15 = 0;
i = 0;

for i =1:w(1)
    if (z2(i, 4) ~= 0)
        x1 = z2(i, 2);  %ow
        x2 = z2(i, 3);  %Column 
        x3 = z2(i, 4);

        h1 = x2 - yaxis_thd; 
        h2 = x2 + yaxis_thd; 

        if(h1 < 1)
            h1 = 1;
        end 
        if(h2 > w(1))
            h2 = w(1);
        end 

        for t1 = h1:h2
            index = (z2(:,3) == t1) & (z2(:,2) == x1);
            matchingRows = z2(index, :);
            f11 = matchingRows(:, 4); 
            if f11 ~= 0;
             q11 = q11+1; 
            end
        end

        for t1 = h1:h2
            index = (z2(:,3) == t1) & (z2(:,2) == x1+1);
            matchingRows = z2(index, :);
            f12 = matchingRows(:, 4); 
            if f12 ~= 0;
             q12 = q12+1; 
            end
        end

        for t1 = h1:h2
            index = (z2(:,3) == t1) & (z2(:,2) == x1-1);
            matchingRows = z2(index, :);
            f14 = matchingRows(:, 4); 
            if f14 ~= 0;
             q14 = q14+1; 
            end
        end

        if (q11 == 1 & q12 == 0 & q14 == 0 )
            z2(i, 4) = 0;
        end
        q11 = 0;
        q12 = 0;
        q13 = 0;        
        q14 = 0;
        q15 = 0;
    end
end

Data_bb = [z2(:,2),z2(:,3),z2(:,4)];
condition = Data_bb(:, 3) == 0;
Data_bb(condition, :) = [];

A=size(Data_bb);
pdd1 = 17.317-0.012*accutimes_1*accutimes_1*accutimes_1+0.394*accutimes_1*accutimes_1-4.214*accutimes_1;
pdd1 = floor(pdd1);

for i = 1:A(1);
    if (mod(Data_bb(i,1), 2) == 0)
        Data_bb(i,2) = (Data_bb(i,2) - pdd1);
    else 
        Data_bb(i,2) = (Data_bb(i,2) + pdd1);
    end
end

% 应用中值滤波
Data_bb(:,3) = medfilt1(Data_bb(:,3), 3);

% 连通区域分析
bw = Data_bb(:,3) > 0;
cc = bwconncomp(bw);
stats = regionprops(cc, 'Area');
idx = find([stats.Area] > 10); % 去除面积小于10的连通区域
bw = ismember(labelmatrix(cc), idx);
Data_bb = Data_bb(bw, :);

% 增加距离阈值判断
% distance_threshold = 20; % 设定距离阈值
noise_points = false(size(Data_bb, 1), 1); % 初始化噪声点标记

for i = 1:size(Data_bb, 1)
    current_point = Data_bb(i, :);
    neighbors = Data_bb(abs(Data_bb(:,1) - current_point(1)) <= yaxis_thd & ...
                abs(Data_bb(:,2) - current_point(2)) <= yaxis_thd, :);
    if size(neighbors, 1) > 1
        distance_diff = abs(neighbors(:,3) - current_point(3));
        if all(distance_diff > distance_threshold)
            noise_points(i) = true; % 标记为噪声点
        end
    end
end

Data_bb(noise_points, :) = []; % 删除噪声点

scatter(Data_bb(:,2), Data_bb(:,1), 15, Data_bb(:,3),'s', 'filled');
clim([min_distance1, max_distance1]);
colorbar;
grid on;

% 设置图形属性和标题
% title('(c)');
xlabel('x-axis');
ylabel('y-axis');
text(66,123, '(c)', 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
set(gca,'FontSize',14, 'Fontname', 'Times New Roman');


xlim([0 134]);
ylim([0 134]);

%颜色设置
% mincolor    = [1 0 0]; % red
% mediancolor = [0 1 0]; % yellow   
% maxcolor    = [0 0 1]; % blue 
ColorMapSize = 100;
int1 = zeros(ColorMapSize,3); 
int2 = zeros(ColorMapSize,3);
for k=1:3
    int1(:,k) = linspace(mincolor(k), mediancolor(k), ColorMapSize);
    int2(:,k) = linspace(mediancolor(k), maxcolor(k), ColorMapSize);
end
meep = [int1(1:end-1,:); int2];
colormap(meep);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% 绘制3D散点图
figure(1); % 创建一个新的图形窗口
subplot(2, 2, 4);
scatter3(Data_bb(:,1), Data_bb(:,2), Data_bb(:,3), 11, Data_bb(:,3), 's', 'filled');
% scatter3(Data_bb(:,1), Data_bb(:,2), Data_bb(:,3), 18, Data_bb(:,3), 's', 'filled');
% scatter3(X, Y, Z, 点大小, 颜色数据, 形状, 'filled')

% 设置颜色映射
colormap(meep); % 使用之前定义的颜色映射
clim([min_distance1, max_distance1]); % 设置colorbar的范围
colorbar; % 添加颜色刻度尺

% 设置图形属性和标题
% title('(d)');
xlabel('x-axis');
ylabel('y-axis');
zlabel('distance'); % 添加Z轴标签
text(66,123, 680,'(d)', 'FontSize', 14, 'VerticalAlignment', 'bottom', 'HorizontalAlignment', 'center');
set(gca,'FontSize',14, 'Fontname', 'Times New Roman');
grid on;

% 设置坐标轴范围
xlim([0 134]);
ylim([0 134]);
zlim([min_distance1, max_distance1]); % 设置Z轴范围

% 其他部分代码保持不变...
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%