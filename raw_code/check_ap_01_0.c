static inline int check_ap(CPUARMState *env, int ap, int domain_prot,

                           int access_type, int is_user)

{

  int prot_ro;



  if (domain_prot == 3) {

    return PAGE_READ | PAGE_WRITE;

  }



  if (access_type == 1)

      prot_ro = 0;

  else

      prot_ro = PAGE_READ;



  switch (ap) {

  case 0:

      if (access_type == 1)

          return 0;

      switch ((env->cp15.c1_sys >> 8) & 3) {

      case 1:

          return is_user ? 0 : PAGE_READ;

      case 2:

          return PAGE_READ;

      default:

          return 0;

      }

  case 1:

      return is_user ? 0 : PAGE_READ | PAGE_WRITE;

  case 2:

      if (is_user)

          return prot_ro;

      else

          return PAGE_READ | PAGE_WRITE;

  case 3:

      return PAGE_READ | PAGE_WRITE;

  case 4: /* Reserved.  */

      return 0;

  case 5:

      return is_user ? 0 : prot_ro;

  case 6:

      return prot_ro;

  case 7:

      if (!arm_feature (env, ARM_FEATURE_V6K))

          return 0;

      return prot_ro;

  default:

      abort();

  }

}
