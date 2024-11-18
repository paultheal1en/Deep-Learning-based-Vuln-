static int hls_decode_entry_wpp(AVCodecContext *avctxt, void *input_ctb_row, int job, int self_id)

{

    HEVCContext *s1  = avctxt->priv_data, *s;

    HEVCLocalContext *lc;

    int ctb_size    = 1<< s1->ps.sps->log2_ctb_size;

    int more_data   = 1;

    int *ctb_row_p    = input_ctb_row;

    int ctb_row = ctb_row_p[job];

    int ctb_addr_rs = s1->sh.slice_ctb_addr_rs + ctb_row * ((s1->ps.sps->width + ctb_size - 1) >> s1->ps.sps->log2_ctb_size);

    int ctb_addr_ts = s1->ps.pps->ctb_addr_rs_to_ts[ctb_addr_rs];

    int thread = ctb_row % s1->threads_number;

    int ret;



    s = s1->sList[self_id];

    lc = s->HEVClc;



    if(ctb_row) {

        ret = init_get_bits8(&lc->gb, s->data + s->sh.offset[ctb_row - 1], s->sh.size[ctb_row - 1]);



        if (ret < 0)

            return ret;

        ff_init_cabac_decoder(&lc->cc, s->data + s->sh.offset[(ctb_row)-1], s->sh.size[ctb_row - 1]);

    }



    while(more_data && ctb_addr_ts < s->ps.sps->ctb_size) {

        int x_ctb = (ctb_addr_rs % s->ps.sps->ctb_width) << s->ps.sps->log2_ctb_size;

        int y_ctb = (ctb_addr_rs / s->ps.sps->ctb_width) << s->ps.sps->log2_ctb_size;



        hls_decode_neighbour(s, x_ctb, y_ctb, ctb_addr_ts);



        ff_thread_await_progress2(s->avctx, ctb_row, thread, SHIFT_CTB_WPP);



        if (avpriv_atomic_int_get(&s1->wpp_err)){

            ff_thread_report_progress2(s->avctx, ctb_row , thread, SHIFT_CTB_WPP);

            return 0;

        }



        ff_hevc_cabac_init(s, ctb_addr_ts);

        hls_sao_param(s, x_ctb >> s->ps.sps->log2_ctb_size, y_ctb >> s->ps.sps->log2_ctb_size);

        more_data = hls_coding_quadtree(s, x_ctb, y_ctb, s->ps.sps->log2_ctb_size, 0);



        if (more_data < 0) {

            s->tab_slice_address[ctb_addr_rs] = -1;



            return more_data;

        }



        ctb_addr_ts++;



        ff_hevc_save_states(s, ctb_addr_ts);

        ff_thread_report_progress2(s->avctx, ctb_row, thread, 1);

        ff_hevc_hls_filters(s, x_ctb, y_ctb, ctb_size);



        if (!more_data && (x_ctb+ctb_size) < s->ps.sps->width && ctb_row != s->sh.num_entry_point_offsets) {



            return 0;

        }



        if ((x_ctb+ctb_size) >= s->ps.sps->width && (y_ctb+ctb_size) >= s->ps.sps->height ) {

            ff_hevc_hls_filter(s, x_ctb, y_ctb, ctb_size);

            ff_thread_report_progress2(s->avctx, ctb_row , thread, SHIFT_CTB_WPP);

            return ctb_addr_ts;

        }

        ctb_addr_rs       = s->ps.pps->ctb_addr_ts_to_rs[ctb_addr_ts];

        x_ctb+=ctb_size;



        if(x_ctb >= s->ps.sps->width) {

            break;

        }

    }




    return 0;

}