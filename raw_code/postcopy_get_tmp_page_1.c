void *postcopy_get_tmp_page(MigrationIncomingState *mis)

{

    if (!mis->postcopy_tmp_page) {

        mis->postcopy_tmp_page = mmap(NULL, getpagesize(),

                             PROT_READ | PROT_WRITE, MAP_PRIVATE |

                             MAP_ANONYMOUS, -1, 0);

        if (!mis->postcopy_tmp_page) {

            error_report("%s: %s", __func__, strerror(errno));

            return NULL;

        }

    }



    return mis->postcopy_tmp_page;

}
