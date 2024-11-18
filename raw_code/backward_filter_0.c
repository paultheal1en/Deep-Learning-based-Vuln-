static void backward_filter(RA288Context *ractx)

{

    float temp1[37]; // RTMP in the spec

    float temp2[11]; // GPTPMP in the spec



    do_hybrid_window(36, 40, 35, ractx->sp_block, temp1, ractx->sp_hist,

                     ractx->sp_rec, syn_window);



    if (!eval_lpc_coeffs(temp1, ractx->sp_lpc, 36))

        colmult(ractx->sp_lpc, ractx->sp_lpc, syn_bw_tab, 36);



    do_hybrid_window(10, 8, 20, ractx->gain_block, temp2, ractx->gain_hist,

                     ractx->gain_rec, gain_window);



    if (!eval_lpc_coeffs(temp2, ractx->gain_lpc, 10))

        colmult(ractx->gain_lpc, ractx->gain_lpc, gain_bw_tab, 10);

}
