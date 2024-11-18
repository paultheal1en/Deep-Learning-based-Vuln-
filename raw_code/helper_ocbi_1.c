void helper_ocbi(CPUSH4State *env, uint32_t address)

{

    memory_content **current = &(env->movcal_backup);

    while (*current)

    {

	uint32_t a = (*current)->address;

	if ((a & ~0x1F) == (address & ~0x1F))

	{

	    memory_content *next = (*current)->next;

            cpu_stl_data(env, a, (*current)->value);

	    

	    if (next == NULL)

	    {

		env->movcal_backup_tail = current;

	    }



	    free (*current);

	    *current = next;

	    break;

	}

    }

}
