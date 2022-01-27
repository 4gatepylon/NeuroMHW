data=importdata('Lab4_ECG_1min_hands_standing_part2_converted.txt');
data_d=detrend(data);
%data_d=lowpass(data_d,1,1000)
f=1000;
s=1/f;
time=(length(data)-1)/f;
timearray=(0:s:time);
plot(timearray,data_d(:,6))
xlim([0 60]);
title('Hand Mounted Standing ECG Graph')
xlabel('Time(s)');
ylabel('mV');