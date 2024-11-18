static int ppc_hash32_check_prot(int prot, int rwx)

{

    int ret;



    if (rwx == 2) {

        if (prot & PAGE_EXEC) {

            ret = 0;

        } else {

            ret = -2;

        }

    } else if (rwx) {

        if (prot & PAGE_WRITE) {

            ret = 0;

        } else {

            ret = -2;

        }

    } else {

        if (prot & PAGE_READ) {

            ret = 0;

        } else {

            ret = -2;

        }

    }



    return ret;

}
