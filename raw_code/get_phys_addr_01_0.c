static inline int get_phys_addr(CPUARMState *env, uint32_t address,

                                int access_type, int is_user,

                                hwaddr *phys_ptr, int *prot,

                                target_ulong *page_size)

{

    /* Fast Context Switch Extension.  */

    if (address < 0x02000000)

        address += env->cp15.c13_fcse;



    if ((env->cp15.c1_sys & 1) == 0) {

        /* MMU/MPU disabled.  */

        *phys_ptr = address;

        *prot = PAGE_READ | PAGE_WRITE | PAGE_EXEC;

        *page_size = TARGET_PAGE_SIZE;

        return 0;

    } else if (arm_feature(env, ARM_FEATURE_MPU)) {

        *page_size = TARGET_PAGE_SIZE;

	return get_phys_addr_mpu(env, address, access_type, is_user, phys_ptr,

				 prot);

    } else if (extended_addresses_enabled(env)) {

        return get_phys_addr_lpae(env, address, access_type, is_user, phys_ptr,

                                  prot, page_size);

    } else if (env->cp15.c1_sys & (1 << 23)) {

        return get_phys_addr_v6(env, address, access_type, is_user, phys_ptr,

                                prot, page_size);

    } else {

        return get_phys_addr_v5(env, address, access_type, is_user, phys_ptr,

                                prot, page_size);

    }

}
