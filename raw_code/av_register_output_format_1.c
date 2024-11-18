void av_register_output_format(AVOutputFormat *format)

{

    AVOutputFormat **p = last_oformat;



    format->next = NULL;

    while(*p || avpriv_atomic_ptr_cas((void * volatile *)p, NULL, format))

        p = &(*p)->next;

    last_oformat = &format->next;

}
