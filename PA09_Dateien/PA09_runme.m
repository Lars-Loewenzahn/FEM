clear all
close all

% Load the original image B1 
imageB1_int = imread('PA09_waves','jpg');
% Save image as a matrix
imageB1 = double(imageB1_int);

% Perform a 2D fast Fourier transformation of B1
fourier = FFT2(imageB1);

% Separate Fourier transform into amplitude and phase
amplitude_B1 = abs(fourier);
phase_B1 = angle(fourier);

% Display the original image B1 (in subplot)
figure;
subplot(2,2,1);
imshow(imageB1, [0,256]); 
%imshow(imageB1_int)
title('Originalbild B1')
% Display the amplitude of the Fourier transform F(B1)
cmin = min(min(amplitude_B1));
cmax = max(max(amplitude_B1));
subplot(2,2,3)
imshow(fftshift(amplitude_B1), [cmin cmax]);
title('Amplitude der Fourier Transf.  F(B1)')
% Display the phase of the Fourier transform F(B1)
dmin = min(min(phase_B1));
dmax = max(max(phase_B1));
subplot(2,2,4)
imshow(fftshift(phase_B1), [dmin dmax]);
title('Phase der Fourier Transf.  F(B1)')

%% F(B1) has only few pixels with a high amplitude: 1 in the center (mean 
%% pixel value of the image B1) and 4 close to the center, corresponding to
%% the frequences of the 2 sinus waves convoluted in B1. 
%%
% Filter the spectrum to keep only the pixels with the biggest  
% and second biggest amplitudes
M1  = max(max(amplitude_B1)); % maximal amplitude
M2  = max(amplitude_B1(amplitude_B1<M1)); % second biggest amplitude
amplitude_B2 = amplitude_B1;
amplitude_B2(amplitude_B2<(1-1e-15)*M2)= 0; % Set all small amplitudes to 0
% Transform the filtered Fourier transform back to the spatial domain
% to create the image B2
imageB2 = ifft2(amplitude_B2.*exp(1i*phase_B1));

% discard noise in the imaginary part
if max(max(imag(imageB2)))<1e-12
    imageB2 = real(imageB2); 
end
% Display B2 and its Fourier transform
figure; 
subplot(2,2,1)
imshow(imageB2, [0, 256]); 
title('Bild B2 (gefiltert aus B1)')
% subplot(2,2,2)
% imshow(imag(imageB2), [0, 256]); 
subplot(2,2,3)
imshow(fftshift(amplitude_B2), [cmin cmax]);
title('Amplitude der Fourier Transf.  F(B2)')
subplot(2,2,4)
imshow(fftshift(phase_B1), [dmin dmax]); 
% F(B1) and F(B2) have the same phase
title('Phase der Fourier Transf.  F(B2)')


% Filter the spectrum to keep only the pixels with the biggest amplitude
% and the thirs biggest amplitude
M3  = max(amplitude_B1(amplitude_B1<M2)); % third biggest amplitude
amplitude_B3 = amplitude_B1;
amplitude_B3(amplitude_B3>(1+1e-15)*M3 & amplitude_B3~=M1)= 0;
amplitude_B3(amplitude_B3<(1-1e-15)*M3)= 0; % Set all amplitudes to 0 
                                            % except the ones corresponding
                                            % to the frequencies of 
                                            % interest
% Transform the filtered Fourier transform back to the spatial domain
% to create the image B2
imageB3 = ifft2(amplitude_B3.*exp(1i*phase_B1));

% discard noise in the imaginary part
if max(max(imag(imageB3)))<1e-12
    imageB3 = real(imageB3); 
end
% Display B2 and its Fourier transform                                                 
figure; 
subplot(2,2,1)
imshow(imageB3, [0, 256]); 
title('Bild B3 (gefiltert aus B1)')
% subplot(2,2,2)
% imshow(imag(imageB3), [0, 256]); 
subplot(2,2,3)
imshow(fftshift(amplitude_B3), [cmin cmax]);
title('Amplitude der Fourier Transf.  F(B3)')
subplot(2,2,4)
imshow(fftshift(phase_B1), [dmin dmax]);
% F(B1) and F(B3) have the same phase
title('Phase der Fourier Transf.  F(B3)')


