static int ram_block_enable_notify(const char *block_name, void *host_addr,

                                   ram_addr_t offset, ram_addr_t length,

                                   void *opaque)

{

    MigrationIncomingState *mis = opaque;

    struct uffdio_register reg_struct;



    reg_struct.range.start = (uintptr_t)host_addr;

    reg_struct.range.len = length;

    reg_struct.mode = UFFDIO_REGISTER_MODE_MISSING;



    /* Now tell our userfault_fd that it's responsible for this area */

    if (ioctl(mis->userfault_fd, UFFDIO_REGISTER, &reg_struct)) {

        error_report("%s userfault register: %s", __func__, strerror(errno));









    return 0;
