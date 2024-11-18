static int encode_residual(FlacEncodeContext *ctx, int ch)

{

    int i, n;

    int min_order, max_order, opt_order, precision, omethod;

    int min_porder, max_porder;

    FlacFrame *frame;

    FlacSubframe *sub;

    int32_t coefs[MAX_LPC_ORDER][MAX_LPC_ORDER];

    int shift[MAX_LPC_ORDER];

    int32_t *res, *smp;



    frame = &ctx->frame;

    sub = &frame->subframes[ch];

    res = sub->residual;

    smp = sub->samples;

    n = frame->blocksize;



    /* CONSTANT */

    for(i=1; i<n; i++) {

        if(smp[i] != smp[0]) break;

    }

    if(i == n) {

        sub->type = sub->type_code = FLAC_SUBFRAME_CONSTANT;

        res[0] = smp[0];

        return sub->obits;

    }



    /* VERBATIM */

    if(n < 5) {

        sub->type = sub->type_code = FLAC_SUBFRAME_VERBATIM;

        encode_residual_verbatim(res, smp, n);

        return sub->obits * n;

    }



    min_order = ctx->options.min_prediction_order;

    max_order = ctx->options.max_prediction_order;

    min_porder = ctx->options.min_partition_order;

    max_porder = ctx->options.max_partition_order;

    precision = ctx->options.lpc_coeff_precision;

    omethod = ctx->options.prediction_order_method;



    /* FIXED */

    if(!ctx->options.use_lpc || max_order == 0 || (n <= max_order)) {

        uint32_t bits[MAX_FIXED_ORDER+1];

        if(max_order > MAX_FIXED_ORDER) max_order = MAX_FIXED_ORDER;

        opt_order = 0;

        bits[0] = UINT32_MAX;

        for(i=min_order; i<=max_order; i++) {

            encode_residual_fixed(res, smp, n, i);

            bits[i] = calc_rice_params_fixed(&sub->rc, min_porder, max_porder, res,

                                             n, i, sub->obits);

            if(bits[i] < bits[opt_order]) {

                opt_order = i;

            }

        }

        sub->order = opt_order;

        sub->type = FLAC_SUBFRAME_FIXED;

        sub->type_code = sub->type | sub->order;

        if(sub->order != max_order) {

            encode_residual_fixed(res, smp, n, sub->order);

            return calc_rice_params_fixed(&sub->rc, min_porder, max_porder, res, n,

                                          sub->order, sub->obits);

        }

        return bits[sub->order];

    }



    /* LPC */

    opt_order = ff_lpc_calc_coefs(&ctx->dsp, smp, n, max_order, precision, coefs,

                               shift, ctx->options.use_lpc, omethod, MAX_LPC_SHIFT, 0);



    if(omethod == ORDER_METHOD_2LEVEL ||

       omethod == ORDER_METHOD_4LEVEL ||

       omethod == ORDER_METHOD_8LEVEL) {

        int levels = 1 << omethod;

        uint32_t bits[levels];

        int order;

        int opt_index = levels-1;

        opt_order = max_order-1;

        bits[opt_index] = UINT32_MAX;

        for(i=levels-1; i>=0; i--) {

            order = min_order + (((max_order-min_order+1) * (i+1)) / levels)-1;

            if(order < 0) order = 0;

            encode_residual_lpc(res, smp, n, order+1, coefs[order], shift[order]);

            bits[i] = calc_rice_params_lpc(&sub->rc, min_porder, max_porder,

                                           res, n, order+1, sub->obits, precision);

            if(bits[i] < bits[opt_index]) {

                opt_index = i;

                opt_order = order;

            }

        }

        opt_order++;

    } else if(omethod == ORDER_METHOD_SEARCH) {

        // brute-force optimal order search

        uint32_t bits[MAX_LPC_ORDER];

        opt_order = 0;

        bits[0] = UINT32_MAX;

        for(i=min_order-1; i<max_order; i++) {

            encode_residual_lpc(res, smp, n, i+1, coefs[i], shift[i]);

            bits[i] = calc_rice_params_lpc(&sub->rc, min_porder, max_porder,

                                           res, n, i+1, sub->obits, precision);

            if(bits[i] < bits[opt_order]) {

                opt_order = i;

            }

        }

        opt_order++;

    } else if(omethod == ORDER_METHOD_LOG) {

        uint32_t bits[MAX_LPC_ORDER];

        int step;



        opt_order= min_order - 1 + (max_order-min_order)/3;

        memset(bits, -1, sizeof(bits));



        for(step=16 ;step; step>>=1){

            int last= opt_order;

            for(i=last-step; i<=last+step; i+= step){

                if(i<min_order-1 || i>=max_order || bits[i] < UINT32_MAX)

                    continue;

                encode_residual_lpc(res, smp, n, i+1, coefs[i], shift[i]);

                bits[i] = calc_rice_params_lpc(&sub->rc, min_porder, max_porder,

                                            res, n, i+1, sub->obits, precision);

                if(bits[i] < bits[opt_order])

                    opt_order= i;

            }

        }

        opt_order++;

    }



    sub->order = opt_order;

    sub->type = FLAC_SUBFRAME_LPC;

    sub->type_code = sub->type | (sub->order-1);

    sub->shift = shift[sub->order-1];

    for(i=0; i<sub->order; i++) {

        sub->coefs[i] = coefs[sub->order-1][i];

    }

    encode_residual_lpc(res, smp, n, sub->order, sub->coefs, sub->shift);

    return calc_rice_params_lpc(&sub->rc, min_porder, max_porder, res, n, sub->order,

                                sub->obits, precision);

}
