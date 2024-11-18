void av_log_default_callback(void* ptr, int level, const char* fmt, va_list vl)

{

    static int print_prefix=1;

    static int count;

    static char line[1024], prev[1024];

    static int is_atty;

    AVClass* avc= ptr ? *(AVClass**)ptr : NULL;

    if(level>av_log_level)

        return;

    line[0]=0;

#undef fprintf

    if(print_prefix && avc) {

        if (avc->parent_log_context_offset) {

            AVClass** parent= *(AVClass***)(((uint8_t*)ptr) + avc->parent_log_context_offset);

            if(parent && *parent){

                snprintf(line, sizeof(line), "[%s @ %p] ", (*parent)->item_name(parent), parent);

            }

        }

        snprintf(line + strlen(line), sizeof(line) - strlen(line), "[%s @ %p] ", avc->item_name(ptr), ptr);

    }



    vsnprintf(line + strlen(line), sizeof(line) - strlen(line), fmt, vl);



    print_prefix= line[strlen(line)-1] == '\n';



#if HAVE_ISATTY

    if(!is_atty) is_atty= isatty(2) ? 1 : -1;

#endif



    if(print_prefix && (flags & AV_LOG_SKIP_REPEATED) && !strcmp(line, prev)){

        count++;

        if(is_atty==1)

            fprintf(stderr, "    Last message repeated %d times\r", count);

        return;

    }

    if(count>0){

        fprintf(stderr, "    Last message repeated %d times\n", count);

        count=0;

    }

    colored_fputs(av_clip(level>>3, 0, 6), line);

    strcpy(prev, line);

}
