% written by debingzhang
% if you have any questions, please fell free to contact
% debingzhangchina@gmail.com

% version 1.0, 2012.11.16


function [ psnr ] = PSNR( Xfull,Xrecover,missing )

    Xrecover = max(0,Xrecover);
    Xrecover = min(255,Xrecover);
    [m,n,dim] = size(Xrecover);
    MSE = 0;
    for i =1 : dim
        MSE = MSE + norm((Xfull(:,:,i)-Xrecover(:,:,i)).*missing(:,:,i),'fro')^2;
    end
    MSE = MSE/nnz(missing);
    
    psnr = 10*log10(255^2/MSE);
        
        
end

