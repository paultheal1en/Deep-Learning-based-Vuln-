void  helper_evaluate_flags_mcp(void)

{

	uint32_t src;

	uint32_t dst;

	uint32_t res;

	uint32_t flags = 0;



	src = env->cc_src;

	dst = env->cc_dest;

	res = env->cc_result;



	if ((res & 0x80000000L) != 0L)

	{

		flags |= N_FLAG;

		if (((src & 0x80000000L) == 0L)

		    && ((dst & 0x80000000L) == 0L))

		{

			flags |= V_FLAG;

		}

		else if (((src & 0x80000000L) != 0L) &&

			 ((dst & 0x80000000L) != 0L))

		{

			flags |= R_FLAG;

		}

	}

	else

	{

		if (res == 0L)

			flags |= Z_FLAG;

		if (((src & 0x80000000L) != 0L)

		    && ((dst & 0x80000000L) != 0L))

			flags |= V_FLAG;

		if ((dst & 0x80000000L) != 0L

		    || (src & 0x80000000L) != 0L)

			flags |= R_FLAG;

	}



	evaluate_flags_writeback(flags);

}
