static void format_line(void *ptr, int level, const char *fmt, va_list vl,

                        AVBPrint part[3], int *print_prefix, int type[2])

{

    AVClass* avc = ptr ? *(AVClass **) ptr : NULL;

    av_bprint_init(part+0, 0, 1);

    av_bprint_init(part+1, 0, 1);

    av_bprint_init(part+2, 0, 65536);



    if(type) type[0] = type[1] = AV_CLASS_CATEGORY_NA + 16;

    if (*print_prefix && avc) {

        if (avc->parent_log_context_offset) {

            AVClass** parent = *(AVClass ***) (((uint8_t *) ptr) +

                                   avc->parent_log_context_offset);

            if (parent && *parent) {

                av_bprintf(part+0, "[%s @ %p] ",

                         (*parent)->item_name(parent), parent);

                if(type) type[0] = get_category(parent);

            }

        }

        av_bprintf(part+1, "[%s @ %p] ",

                 avc->item_name(ptr), ptr);

        if(type) type[1] = get_category(ptr);

    }



    av_vbprintf(part+2, fmt, vl);



    if(*part[0].str || *part[1].str || *part[2].str) {

        char lastc = part[2].len ? part[2].str[part[2].len - 1] : 0;

        *print_prefix = lastc == '\n' || lastc == '\r';

    }

}
