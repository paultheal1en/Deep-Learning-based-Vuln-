static inline void vmsvga_update_rect_delayed(struct vmsvga_state_s *s,

                int x, int y, int w, int h)

{

    struct vmsvga_rect_s *rect = &s->redraw_fifo[s->redraw_fifo_last ++];

    s->redraw_fifo_last &= REDRAW_FIFO_LEN - 1;

    rect->x = x;

    rect->y = y;

    rect->w = w;

    rect->h = h;

}
