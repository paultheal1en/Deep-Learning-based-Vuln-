void ff_acelp_lspd2lpc(const double *lsp, float *lpc)

{

    double pa[6], qa[6];

    int   i;



    lsp2polyf(lsp,     pa, 5);

    lsp2polyf(lsp + 1, qa, 5);



    for (i=4; i>=0; i--)

    {

        double paf = pa[i+1] + pa[i];

        double qaf = qa[i+1] - qa[i];



        lpc[i  ] = 0.5*(paf+qaf);

        lpc[9-i] = 0.5*(paf-qaf);

    }

}
