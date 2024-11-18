static void handle_char(CCaptionSubContext *ctx, char hi, char lo, int64_t pts)

{

    struct Screen *screen = get_writing_screen(ctx);

    char *row = screen->characters[ctx->cursor_row];

    int ret;



    SET_FLAG(screen->row_used,ctx->cursor_row);



    ret = write_char(ctx, row, ctx->cursor_column, hi);

    if( ret == 0 )

        ctx->cursor_column++;



    if(lo) {

        ret = write_char(ctx, row, ctx->cursor_column, lo);

        if ( ret == 0 )

            ctx->cursor_column++;

    }

    write_char(ctx, row, ctx->cursor_column, 0);



    /* reset prev command since character can repeat */

    ctx->prev_cmd[0] = 0;

    ctx->prev_cmd[1] = 0;

    if (lo)

       av_dlog(ctx, "(%c,%c)\n",hi,lo);

    else

       av_dlog(ctx, "(%c)\n",hi);

}
