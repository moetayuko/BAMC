function [y,S,Yv] = BAMC(Av, classnum, gamma, r)
viewnum = numel(Av);
Fv = cell(1,viewnum);
FFv = cell(1,viewnum);
Av_rep = sparse(0);
for v = 1:viewnum
    Av_rep = Av{v} + Av_rep;
    [Fv{v}, ~] = eigs(Ls(Av{v}), classnum+1, 'smallestreal');
    Fv{v} = Fv{v}(:,2:classnum+1);
    Fv{v} = Fv{v} ./ sqrt(sum(Fv{v}.^2,2));
    Fv{v} = orth(Fv{v});
    FFv{v} = Fv{v} * Fv{v}';
end
Av_rep = Av_rep / viewnum;
L_rep = Ls(Av_rep);
[Yv, ~] = eigs(L_rep, classnum+1, 'smallestreal');
Yv = Yv(:,2:classnum+1);
Yv = Yv ./ sqrt(sum(Yv.^2,2))
Yv = orth(Yv);
lambda = gamma;
NITER_outer = 30;
S = Av_rep;

for iter = 1:NITER_outer
    pv = reweighted_pv(Yv,FFv,r);
    sv = reweighted_sv(S,Av,r);

    distf = squareform(pdist(Yv, 'squaredeuclidean'));
    sum_sv = sum(sv);

    a_temp = sparse(0);
    for v = 1 : viewnum
        a_temp = a_temp + sv(v).*Av{v};
    end
    ad = (a_temp - (lambda/(4*gamma))*distf) / sum_sv;
    S = EProjSimplex_native(ad);

    S = (S+S')/2;
    L = Ls(S);
    Fs = 0;
    for v = 1: viewnum
        Fs = Fs+pv(v)/lambda*FFv{v};
    end
    [Yv, ~] = eigs(L-Fs, classnum, 'smallestreal');

    ev = eigs(L, classnum + 1, 'smallestreal');
    assert(~anynan(ev));
    thre = 1e-11;
    fn1 = sum(ev(1:classnum));
    fn2 = fn1 + ev(end);
    if fn1 > thre
        lambda = 2*lambda;
    elseif fn2 < thre
        lambda = lambda/2;
    else
        break;
    end
end
y = conncomp(graph(S));

if max(y) ~= classnum
    error('Can not find the correct cluster number: %f', gamma)
end
end

function L0 = Ls(A)
    D10 = diag(sum(A));
    L0 = D10 - A;
end

function pv = reweighted_pv(F1, FF2, r)
    viewnum = numel(FF2);
    pv = zeros(1,viewnum);
    FF1 = F1*F1';
    for v = 1:viewnum
        pv(v) = r*norm(FF1 - FF2{v},'fro')^(r-2);
    end
end

function sv = reweighted_sv(S1, S2, r)
    viewnum = numel(S2);
    sv = zeros(1,viewnum);
    for v = 1:viewnum
        sv(v) = 0.5*r*norm(S1-S2{v},'fro')^(r-2);
    end
end
