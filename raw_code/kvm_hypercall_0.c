static long kvm_hypercall(unsigned long nr, unsigned long param1,

                          unsigned long param2)

{

	register ulong r_nr asm("1") = nr;

	register ulong r_param1 asm("2") = param1;

	register ulong r_param2 asm("3") = param2;

	register long retval asm("2");



	asm volatile ("diag 2,4,0x500"

		      : "=d" (retval)

		      : "d" (r_nr), "0" (r_param1), "r"(r_param2)

		      : "memory", "cc");



	return retval;

}
