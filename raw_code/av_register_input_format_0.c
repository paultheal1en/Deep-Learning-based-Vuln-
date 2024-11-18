void av_register_input_format(AVInputFormat *format)

{

    AVInputFormat **p = &first_iformat;



    while (*p != NULL)

        p = &(*p)->next;



    *p = format;

    format->next = NULL;

}
