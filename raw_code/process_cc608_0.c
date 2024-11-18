static int process_cc608(CCaptionSubContext *ctx, int64_t pts, uint8_t hi, uint8_t lo)

{

    int ret = 0;

#define COR3(var, with1, with2, with3)  ( (var) == (with1) ||  (var) == (with2) || (var) == (with3) )

    if ( hi == ctx->prev_cmd[0] && lo == ctx->prev_cmd[1]) {

    /* ignore redundant command */

    } else if ( (hi == 0x10 && (lo >= 0x40 || lo <= 0x5f)) ||

              ( (hi >= 0x11 && hi <= 0x17) && (lo >= 0x40 && lo <= 0x7f) ) ) {

        handle_pac(ctx, hi, lo);

    } else if ( ( hi == 0x11 && lo >= 0x20 && lo <= 0x2f ) ||

                ( hi == 0x17 && lo >= 0x2e && lo <= 0x2f) ) {

        handle_textattr(ctx, hi, lo);

    } else if ( COR3(hi, 0x14, 0x15, 0x1C) && lo == 0x20 ) {

    /* resume caption loading */

        ctx->mode = CCMODE_POPON;

    } else if ( COR3(hi, 0x14, 0x15, 0x1C) && lo == 0x24 ) {

        handle_delete_end_of_row(ctx, hi, lo);

    } else if ( COR3(hi, 0x14, 0x15, 0x1C) && lo == 0x25 ) {

        ctx->rollup = 2;

        ctx->mode = CCMODE_ROLLUP_2;

    } else if ( COR3(hi, 0x14, 0x15, 0x1C) && lo == 0x26 ) {

        ctx->rollup = 3;

        ctx->mode = CCMODE_ROLLUP_3;

    } else if ( COR3(hi, 0x14, 0x15, 0x1C) && lo == 0x27 ) {

        ctx->rollup = 4;

        ctx->mode = CCMODE_ROLLUP_4;

    } else if ( COR3(hi, 0x14, 0x15, 0x1C) && lo == 0x29 ) {

    /* resume direct captioning */

        ctx->mode = CCMODE_PAINTON;

    } else if ( COR3(hi, 0x14, 0x15, 0x1C) && lo == 0x2B ) {

    /* resume text display */

        ctx->mode = CCMODE_TEXT;

    } else if ( COR3(hi, 0x14, 0x15, 0x1C) && lo == 0x2C ) {

    /* erase display memory */

        ret = handle_edm(ctx, pts);

    } else if ( COR3(hi, 0x14, 0x15, 0x1C) && lo == 0x2D ) {

    /* carriage return */

        av_dlog(ctx, "carriage return\n");

        reap_screen(ctx, pts);

        roll_up(ctx);

        ctx->screen_changed = 1;

        ctx->cursor_column = 0;

    } else if ( COR3(hi, 0x14, 0x15, 0x1C) && lo == 0x2F ) {

    /* end of caption */

        av_dlog(ctx, "handle_eoc\n");

        ret = handle_eoc(ctx, pts);

    } else if (hi>=0x20) {

    /* Standard characters (always in pairs) */

        handle_char(ctx, hi, lo, pts);

    } else {

    /* Ignoring all other non data code */

        av_dlog(ctx, "Unknown command 0x%hhx 0x%hhx\n", hi, lo);

    }



    /* set prev command */

     ctx->prev_cmd[0] = hi;

     ctx->prev_cmd[1] = lo;



#undef COR3

    return ret;



}
