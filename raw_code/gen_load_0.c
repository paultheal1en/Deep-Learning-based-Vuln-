static void gen_load(DisasContext *dc, TCGv dst, TCGv addr, 

		     unsigned int size, int sign)

{

	int mem_index = cpu_mmu_index(dc->env);



	/* If we get a fault on a delayslot we must keep the jmp state in

	   the cpu-state to be able to re-execute the jmp.  */

	if (dc->delayed_branch == 1)

		cris_store_direct_jmp(dc);



	if (size == 1) {

		if (sign)

			tcg_gen_qemu_ld8s(dst, addr, mem_index);

		else

			tcg_gen_qemu_ld8u(dst, addr, mem_index);

	}

	else if (size == 2) {

		if (sign)

			tcg_gen_qemu_ld16s(dst, addr, mem_index);

		else

			tcg_gen_qemu_ld16u(dst, addr, mem_index);

	}

	else if (size == 4) {

		tcg_gen_qemu_ld32u(dst, addr, mem_index);

	}

	else if (size == 8) {

		tcg_gen_qemu_ld64(dst, addr, mem_index);

	}

}
