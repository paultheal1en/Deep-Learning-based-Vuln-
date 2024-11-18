static void filter_edges(void *dst1, void *prev1, void *cur1, void *next1,

                         int w, int prefs, int mrefs, int parity, int mode)

{

    uint8_t *dst  = dst1;

    uint8_t *prev = prev1;

    uint8_t *cur  = cur1;

    uint8_t *next = next1;

    int x;

    uint8_t *prev2 = parity ? prev : cur ;

    uint8_t *next2 = parity ? cur  : next;



    /* Only edge pixels need to be processed here.  A constant value of false

     * for is_not_edge should let the compiler ignore the whole branch. */

    FILTER(0, 3, 0)



    dst  = (uint8_t*)dst1  + w - 3;

    prev = (uint8_t*)prev1 + w - 3;

    cur  = (uint8_t*)cur1  + w - 3;

    next = (uint8_t*)next1 + w - 3;

    prev2 = (uint8_t*)(parity ? prev : cur);

    next2 = (uint8_t*)(parity ? cur  : next);



    FILTER(w - 3, w, 0)

}
