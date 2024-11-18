static const void *boston_kernel_filter(void *opaque, const void *kernel,

                                        hwaddr *load_addr, hwaddr *entry_addr)

{

    BostonState *s = BOSTON(opaque);



    s->kernel_entry = *entry_addr;



    return kernel;

}
