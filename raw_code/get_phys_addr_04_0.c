static inline int get_phys_addr(CPUARMState *env, target_ulong address,

                                int access_type, int is_user,

                                hwaddr *phys_ptr, int *prot,

                                target_ulong *page_size)

{

    /* This is not entirely correct as get_phys_addr() can also be called

     * from ats_write() for an address translation of a specific regime.

     */

    uint32_t sctlr = A32_BANKED_CURRENT_REG_GET(env, sctlr);



    /* Fast Context Switch Extension.  */

    if (address < 0x02000000)

        address += env->cp15.c13_fcse;



    if ((sctlr & SCTLR_M) == 0) {

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

    } else if (sctlr & SCTLR_XP) {

        return get_phys_addr_v6(env, address, access_type, is_user, phys_ptr,

                                prot, page_size);

    } else {

        return get_phys_addr_v5(env, address, access_type, is_user, phys_ptr,

                                prot, page_size);

    }

}
