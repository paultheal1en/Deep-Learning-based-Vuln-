static void coded_frame_add(void *list, struct FrameListData *cx_frame)

{

    struct FrameListData **p = list;



    while (*p != NULL)

        p = &(*p)->next;

    *p = cx_frame;

    cx_frame->next = NULL;

}
