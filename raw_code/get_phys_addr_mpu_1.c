static int get_phys_addr_mpu(CPUARMState *env, uint32_t address,

                             int access_type, int is_user,

                             hwaddr *phys_ptr, int *prot)

{

    int n;

    uint32_t mask;

    uint32_t base;



    *phys_ptr = address;

    for (n = 7; n >= 0; n--) {

	base = env->cp15.c6_region[n];

	if ((base & 1) == 0)

	    continue;

	mask = 1 << ((base >> 1) & 0x1f);

	/* Keep this shift separate from the above to avoid an

	   (undefined) << 32.  */

	mask = (mask << 1) - 1;

	if (((base ^ address) & ~mask) == 0)

	    break;

    }

    if (n < 0)

	return 2;



    if (access_type == 2) {

	mask = env->cp15.c5_insn;

    } else {

	mask = env->cp15.c5_data;

    }

    mask = (mask >> (n * 4)) & 0xf;

    switch (mask) {

    case 0:

	return 1;

    case 1:

	if (is_user)

	  return 1;

	*prot = PAGE_READ | PAGE_WRITE;

	break;

    case 2:

	*prot = PAGE_READ;

	if (!is_user)

	    *prot |= PAGE_WRITE;

	break;

    case 3:

	*prot = PAGE_READ | PAGE_WRITE;

	break;

    case 5:

	if (is_user)

	    return 1;

	*prot = PAGE_READ;

	break;

    case 6:

	*prot = PAGE_READ;

	break;

    default:

	/* Bad permission.  */

	return 1;

    }

    *prot |= PAGE_EXEC;

    return 0;

}
