static inline void vmsvga_update_rect_flush(struct vmsvga_state_s *s)

{

    struct vmsvga_rect_s *rect;

    if (s->invalidated) {

        s->redraw_fifo_first = s->redraw_fifo_last;

        return;

    }

    /* Overlapping region updates can be optimised out here - if someone

     * knows a smart algorithm to do that, please share.  */

    while (s->redraw_fifo_first != s->redraw_fifo_last) {

        rect = &s->redraw_fifo[s->redraw_fifo_first ++];

        s->redraw_fifo_first &= REDRAW_FIFO_LEN - 1;

        vmsvga_update_rect(s, rect->x, rect->y, rect->w, rect->h);

    }

}
