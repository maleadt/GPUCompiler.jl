# implementation of the GPUCompiler interfaces for generating Metal code

const Metal_LLVM_Tools_jll = LazyModule("Metal_LLVM_Tools_jll", UUID("0418c028-ff8c-56b8-a53e-0f9676ed36fc"))

## target

export MetalCompilerTarget

Base.@kwdef struct MetalCompilerTarget <: AbstractCompilerTarget
    macos::VersionNumber
end

function Base.hash(target::MetalCompilerTarget, h::UInt)
    hash(target.macos, h)
end

source_code(target::MetalCompilerTarget) = "metal"

# Metal is not supported by our LLVM builds, so we can't get a target machine
llvm_machine(::MetalCompilerTarget) = nothing

llvm_triple(target::MetalCompilerTarget) = "air64-apple-macosx$(target.macos)"

llvm_datalayout(target::MetalCompilerTarget) =
    "e-p:64:64:64"*
    "-i1:8:8-i8:8:8-i16:16:16-i32:32:32-i64:64:64"*
    "-f32:32:32-f64:64:64"*
    "-v16:16:16-v24:32:32-v32:32:32-v48:64:64-v64:64:64-v96:128:128-v128:128:128-v192:256:256-v256:256:256-v512:512:512-v1024:1024:1024"*
    "-n8:16:32"

needs_byval(job::CompilerJob{MetalCompilerTarget}) = false

## job

# TODO: encode debug build or not in the compiler job
#       https://github.com/JuliaGPU/CUDAnative.jl/issues/368
runtime_slug(job::CompilerJob{MetalCompilerTarget}) = "metal-macos$(job.target.macos)"

isintrinsic(@nospecialize(job::CompilerJob{MetalCompilerTarget}), fn::String) =
    return startswith(fn, "air.")

const LLVMMETALFUNCCallConv   = LLVM.API.LLVMCallConv(102)
const LLVMMETALKERNELCallConv = LLVM.API.LLVMCallConv(103)

const metal_struct_names = [:MtlDeviceArray, :MtlDeviceMatrix, :MtlDeviceVector]

# Initial mapping of types - There has to be a better way
const jl_type_to_c = Dict(
                    Float32 => "float",
                    Float16 => "half",
                    Int64   => "long",
                    UInt64  => "ulong",
                    Int32   => "int",
                    UInt32  => "uint",
                    Int16   => "short",
                    UInt16  => "ushort",
                    Int8    => "char",
                    UInt8   => "uchar",
                    Bool    => "char" # xxx: ?
                )

function process_module!(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module)
    # calling convention
    for f in functions(mod)
        #callconv!(f, LLVMMETALFUNCCallConv)
        # XXX: this makes InstCombine erase kernel->func calls.
        #      do we even need this? why?
    end
end

function process_entry!(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(process_entry!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    if job.source.kernel
        # calling convention
        callconv!(entry, LLVMMETALKERNELCallConv)
    end

    return entry
end

# TODO: why is this done in finish_module? maybe just in process_entry?
function finish_module!(@nospecialize(job::CompilerJob{MetalCompilerTarget}), mod::LLVM.Module, entry::LLVM.Function)
    entry = invoke(finish_module!, Tuple{CompilerJob, LLVM.Module, LLVM.Function}, job, mod, entry)

    ctx = context(mod)
    entry_fn = LLVM.name(entry)

    if job.source.kernel
        # Change intrinsics to be input arguments as necessary and add metadata
        arguments = add_input_arguments!(job, mod, entry)

        # Alter air intrinsic names
        for func in collect(LLVM.functions(mod))
            !startswith(name(func), "julia.air") && continue
            LLVM.name!(func, name(func)[7:end])
        end
        entry = LLVM.functions(mod)[entry_fn]
        add_metadata!(job, mod, entry, arguments)

        # TESTING: Adding llvm.module.flags
        # wchar_size = 4
        # wchar_key = "wchar_size"
        wchar_md = Metadata[]
        push!(wchar_md, Metadata(ConstantInt(Int32(1); ctx)))
        push!(wchar_md, MDString("wchar_size"; ctx))
        push!(wchar_md, Metadata(ConstantInt(Int32(4); ctx)))
        wchar_md = MDNode(wchar_md; ctx)
        push!(metadata(mod)["llvm.module.flags"], wchar_md)

        # LLVM.API.LLVMAddModuleFlag(mod, LLVM.API.LLVMModuleFlagBehavior(1),
        #         Cstring(pointer(wchar_key)), Csize_t(length(wchar_key)),
        #         wchar_md)

        # !4 = !{i32 7, !"air.max_device_buffers", i32 31}
        max_buff = Metadata[]
        max_buff_key = "air.max_device_buffers"
        push!(max_buff, Metadata(ConstantInt(Int32(7); ctx)))
        push!(max_buff, MDString("air.max_device_buffers"; ctx))
        push!(max_buff, Metadata(ConstantInt(Int32(31); ctx)))
        max_buff = MDNode(max_buff; ctx)
        push!(metadata(mod)["llvm.module.flags"], max_buff)

        # LLVM.API.LLVMAddModuleFlag(mod, LLVM.API.LLVMModuleFlagBehavior(7),
        #         Cstring(pointer(max_buff_key)), Csize_t(length(max_buff_key)),
        #         max_buff)

        # !5 = !{i32 7, !"air.max_constant_buffers", i32 31}
        max_const_buff_md = Metadata[]
        push!(max_const_buff_md, Metadata(ConstantInt(Int32(7); ctx)))
        push!(max_const_buff_md, MDString("air.max_constant_buffers"; ctx))
        push!(max_const_buff_md, Metadata(ConstantInt(Int32(31); ctx)))
        max_const_buff_md = MDNode(max_const_buff_md; ctx)
        push!(metadata(mod)["llvm.module.flags"], max_const_buff_md)

        # !6 = !{i32 7, !"air.max_threadgroup_buffers", i32 31}
        max_threadgroup_buff_md = Metadata[]
        push!(max_threadgroup_buff_md, Metadata(ConstantInt(Int32(7); ctx)))
        push!(max_threadgroup_buff_md, MDString("air.max_threadgroup_buffers"; ctx))
        push!(max_threadgroup_buff_md, Metadata(ConstantInt(Int32(31); ctx)))
        max_threadgroup_buff_md = MDNode(max_threadgroup_buff_md; ctx)
        push!(metadata(mod)["llvm.module.flags"], max_threadgroup_buff_md)

        # !7 = !{i32 7, !"air.max_textures", i32 128}
        max_textures_md = Metadata[]
        push!(max_textures_md, Metadata(ConstantInt(Int32(7); ctx)))
        push!(max_textures_md, MDString("air.max_textures"; ctx))
        push!(max_textures_md, Metadata(ConstantInt(Int32(128); ctx)))
        max_textures_md = MDNode(max_textures_md; ctx)
        push!(metadata(mod)["llvm.module.flags"], max_textures_md)

        # !8 = !{i32 7, !"air.max_read_write_textures", i32 8}
        max_rw_textures_md = Metadata[]
        push!(max_rw_textures_md, Metadata(ConstantInt(Int32(7); ctx)))
        push!(max_rw_textures_md, MDString("air.max_read_write_textures"; ctx))
        push!(max_rw_textures_md, Metadata(ConstantInt(Int32(8); ctx)))
        max_rw_textures_md = MDNode(max_rw_textures_md; ctx)
        push!(metadata(mod)["llvm.module.flags"], max_rw_textures_md)

        # !9 = !{i32 7, !"air.max_samplers", i32 16}
        max_samplers_md = Metadata[]
        push!(max_samplers_md, Metadata(ConstantInt(Int32(7); ctx)))
        push!(max_samplers_md, MDString("air.max_samplers"; ctx))
        push!(max_samplers_md, Metadata(ConstantInt(Int32(16); ctx)))
        max_samplers_md = MDNode(max_samplers_md; ctx)
        push!(metadata(mod)["llvm.module.flags"], max_samplers_md)


        # function LLVMAddModuleFlag(M, Behavior, Key, KeyLen, Val)
        #     ccall((:LLVMAddModuleFlag, libllvm[]), Cvoid, (LLVMModuleRef, LLVMModuleFlagBehavior, Cstring, Csize_t, LLVMMetadataRef), M, Behavior, Key, KeyLen, Val)
        # end

        # Add llvm.ident
        # !llvm.ident = !{!10}
        # !10 = !{!"Apple metal version 31001.363 (metalfe-31001.363)"}
        llvm_ident_md = Metadata[]
        push!(llvm_ident_md, MDString("Apple metal version 31001.363 (metalfe-31001.363)"; ctx))
        llvm_ident_md = MDNode(llvm_ident_md; ctx)
        push!(metadata(mod)["llvm.ident"], llvm_ident_md)


        # Add air version metadata
        air_md = Metadata[]
        push!(air_md, Metadata(ConstantInt(Int32(2); ctx)))
        push!(air_md, Metadata(ConstantInt(Int32(4); ctx)))
        push!(air_md, Metadata(ConstantInt(Int32(0); ctx)))
        air_md = MDNode(air_md; ctx)
        push!(metadata(mod)["air.version"], air_md)

        # !air.language_version = !{!12}
        # !12 = !{!"Metal", i32 2, i32 4, i32 0}
        air_lang_md = Metadata[]
        push!(air_lang_md, MDString("Metal"; ctx))
        push!(air_lang_md, Metadata(ConstantInt(Int32(2); ctx)))
        push!(air_lang_md, Metadata(ConstantInt(Int32(4); ctx)))
        push!(air_lang_md, Metadata(ConstantInt(Int32(0); ctx)))
        air_lang_md = MDNode(air_lang_md; ctx)
        push!(metadata(mod)["air.language_version"], air_lang_md)

    end

    return functions(mod)[entry_fn]
end

const kernel_intrinsics = Dict()
for intr in [
        "dispatch_quadgroups_per_threadgroup", "dispatch_simdgroups_per_threadgroup",
        "quadgroup_index_in_threadgroup", "quadgroups_per_threadgroup",
        "simdgroup_index_in_threadgroup", "simdgroups_per_threadgroup",
        "thread_index_in_quadgroup", "thread_index_in_simdgroup", "thread_index_in_threadgroup",
        "thread_execution_width", "threads_per_simdgroup"],
    (intr_typ, air_typ, julia_typ) in [
        ("i32",   "uint",  UInt32),
        ("i16",   "ushort",  UInt16),
    ]
    push!(kernel_intrinsics,
          "julia.air.$intr.$intr_typ" =>
          (air_intr="$intr.$air_typ", air_typ, air_name=intr, julia_typ))
end
for intr in [
        "dispatch_threads_per_threadgroup",
        "grid_origin", "grid_size",
        "thread_position_in_grid", "thread_position_in_threadgroup",
        "threadgroup_position_in_grid", "threadgroups_per_grid",
        "threads_per_grid", "threads_per_threadgroup"],
    (intr_typ, air_typ, julia_typ) in [
        ("i32",   "uint",  UInt32),
        ("v2i32", "uint2", NTuple{2, VecElement{UInt32}}),
        ("v3i32", "uint3", NTuple{3, VecElement{UInt32}}),
        ("i16",   "ushort",  UInt16),
        ("v2i16", "ushort2", NTuple{2, VecElement{UInt16}}),
        ("v3i16", "ushort3", NTuple{3, VecElement{UInt16}}),
    ]
    push!(kernel_intrinsics,
          "julia.air.$intr.$intr_typ" =>
          (air_intr="$intr.$air_typ", air_typ, air_name=intr, julia_typ))
end

function add_input_arguments!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                              entry::LLVM.Function)
    ctx = context(mod)
    entry_fn = LLVM.name(entry)

    # figure out which intrinsics are used and need to be added as arguments
    used_intrinsics = filter(keys(kernel_intrinsics)) do intr_fn
        haskey(functions(mod), intr_fn)
    end |> collect
    # TODO: Figure out how to not be inefficient with these changes
    nargs = length(used_intrinsics)

    # determine which functions need these arguments
    worklist = Set{LLVM.Function}([entry])
    for intr_fn in used_intrinsics
        push!(worklist, functions(mod)[intr_fn])
    end
    worklist_length = 0
    while worklist_length != length(worklist)
        # iteratively discover functions that use an intrinsic or any function calling it
        worklist_length = length(worklist)
        additions = LLVM.Function[]
        for f in worklist, use in uses(f)
            inst = user(use)::Instruction
            bb = LLVM.parent(inst)
            new_f = LLVM.parent(bb)
            in(new_f, worklist) || push!(additions, new_f)
        end
        for f in additions
            push!(worklist, f)
        end
    end
    for intr_fn in used_intrinsics
        delete!(worklist, functions(mod)[intr_fn])
    end

    # add the arguments
    # NOTE: we could be more fine-grained, only adding the specific intrinsics used by this function.
    #       not sure if that's worth it though.
    workmap = Dict{LLVM.Function, LLVM.Function}()
    for f in worklist
        fn = LLVM.name(f)
        ft = eltype(llvmtype(f))
        LLVM.name!(f, fn * ".orig")
        # create a new function
        new_param_types = LLVMType[]

        # Alter LLVM representation of Metal device arrays and Core.LLVMPtrs
        for (i, param) in enumerate(job.source.tt.parameters)
            # Alter MtlDeviceArrays to have correct addresspace
            if param.name.name in metal_struct_names
                ## Unnamed struct - Named structs give the metadata generation trouble
                elems = collect(elements(convert(LLVMType, param; ctx)))
                elems[1] = LLVM.PointerType(convert(LLVM.LLVMType, param.parameters[1]; ctx), param.parameters[3])
                struct_typ = LLVM.StructType(elems; ctx)
                # Alter addresspace of struct type to match Metal device array
                param_typ = LLVM.PointerType(struct_typ, param.parameters[3])
                push!(new_param_types, param_typ)

            # Give opaque LLVMPtrs the appropriate type
            elseif param <: Core.LLVMPtr
                param_typ = LLVM.PointerType(convert(LLVMType, param.parameters[1]; ctx), param.parameters[2])
                push!(new_param_types, param_typ)

            # Don't alter any other argument types
            else
                push!(new_param_types, parameters(ft)[i])
            end
        end

        for intr_fn in used_intrinsics
            llvm_typ = convert(LLVMType, kernel_intrinsics[intr_fn].julia_typ; ctx)
            push!(new_param_types, llvm_typ)
        end
        new_ft = LLVM.FunctionType(return_type(ft), new_param_types)
        new_f = LLVM.Function(mod, fn, new_ft)
        linkage!(new_f, linkage(f))
        for (arg, new_arg) in zip(parameters(f), parameters(new_f))
            LLVM.name!(new_arg, LLVM.name(arg))
        end
        for (intr_fn, new_arg) in zip(used_intrinsics, parameters(new_f)[end-nargs+1:end])
            LLVM.name!(new_arg, kernel_intrinsics[intr_fn].air_name)
        end

        workmap[f] = new_f
    end

    # clone and rewrite the function bodies.
    # we don't need to rewrite much as the arguments are added last.
    for (f, new_f) in workmap
        # use a value mapper for rewriting function arguments
        value_map = Dict{LLVM.Value, LLVM.Value}()
        for (param, new_param) in zip(parameters(f), parameters(new_f))
            LLVM.name!(new_param, LLVM.name(param))
            value_map[param] = new_param
        end

        # use a value materializer for replacing uses of the function in constants
        function materializer(val)
            opcodes = (LLVM.API.LLVMPtrToInt, LLVM.API.LLVMAddrSpaceCast, LLVM.API.LLVMBitCast)
            if val isa LLVM.ConstantExpr && opcode(val) in opcodes
                src = operands(val)[1]
                if haskey(workmap, src)
                    return if opcode(val) == LLVM.API.LLVMPtrToInt
                        LLVM.const_ptrtoint(workmap[src], llvmtype(val))
                    elseif opcode(val) == LLVM.API.LLVMAddrSpaceCast
                        LLVM.const_addrspacecast(workmap[src], llvmtype(val))
                    elseif opcode(val) == LLVM.API.LLVMBitCast
                        LLVM.const_bitcast(workmap[src], llvmtype(val))
                    end
                end
            end
            return val
        end

        # we don't want module-level changes, because otherwise LLVM will clone metadata,
        # resulting in mismatching references between `!dbg` metadata and `dbg` instructions
        clone_into!(new_f, f; value_map, materializer,
                    changes=LLVM.API.LLVMCloneFunctionChangeTypeLocalChangesOnly)

        # we can't remove this function yet, as we might still need to rewrite any called,
        # but remove the IR already
        empty!(f)
    end

    # drop unused constants that may be referring to the old functions
    # XXX: can we do this differently?
    for f in worklist
        for use in uses(f)
            val = user(use)
            if val isa LLVM.ConstantExpr && isempty(uses(val))
                LLVM.unsafe_destroy!(val)
            end
        end
    end

    # update other uses of the old function, modifying call sites to pass the arguments
    function rewrite_uses!(f, new_f)
        # update uses
        Builder(ctx) do builder
            for use in uses(f)
                val = user(use)
                if val isa LLVM.CallInst || val isa LLVM.InvokeInst || val isa LLVM.CallBrInst
                    callee_f = LLVM.parent(LLVM.parent(val))
                    # forward the arguments
                    position!(builder, val)
                    new_val = if val isa LLVM.CallInst
                        call!(builder, new_f, [arguments(val)..., parameters(callee_f)[end-nargs+1:end]...], operand_bundles(val))
                    else
                        # TODO: invoke and callbr
                        error("Rewrite of $(typeof(val))-based calls is not implemented: $val")
                    end
                    callconv!(new_val, callconv(val))

                    replace_uses!(val, new_val)
                    @assert isempty(uses(val))
                    unsafe_delete!(LLVM.parent(val), val)
                elseif val isa LLVM.ConstantExpr && opcode(val) == LLVM.API.LLVMBitCast
                    # XXX: why isn't this caught by the value materializer above?
                    target = operands(val)[1]
                    @assert target == f
                    new_val = LLVM.const_bitcast(new_f, llvmtype(val))
                    rewrite_uses!(val, new_val)
                    # we can't simply replace this constant expression, as it may be used
                    # as a call, taking arguments (so we need to rewrite it to pass the input arguments)

                    # drop the old constant if it is unused
                    # XXX: can we do this differently?
                    if isempty(uses(val))
                        LLVM.unsafe_destroy!(val)
                    end
                else
                    error("Cannot rewrite unknown use of function: $val")
                end
            end
        end
    end
    for (f, new_f) in workmap
        rewrite_uses!(f, new_f)
        @assert isempty(uses(f))
        unsafe_delete!(mod, f)
    end

    # replace uses of the intrinsics with references to the input arguments
    for (i, intr_fn) in enumerate(used_intrinsics)
        intr = functions(mod)[intr_fn]
        for use in uses(intr)
            val = user(use)
            callee_f = LLVM.parent(LLVM.parent(val))
            if val isa LLVM.CallInst || val isa LLVM.InvokeInst || val isa LLVM.CallBrInst
                replace_uses!(val, parameters(callee_f)[end-nargs+i])
            else
                error("Cannot rewrite unknown use of function: $val")
            end

            @assert isempty(uses(val))
            unsafe_delete!(LLVM.parent(val), val)
        end
        @assert isempty(uses(intr))
        unsafe_delete!(mod, intr)
    end

    return used_intrinsics
end

function add_metadata!(@nospecialize(job::CompilerJob), mod::LLVM.Module,
                       entry::LLVM.Function, used_intrinsics::Vector)
    ctx = context(mod)

    # Recursively generate metadata for normal kernel arguments
    function add_md(arg, field_info=nothing, level=1)
        structinfo(T,i) = (fieldoffset(T,i), fieldname(T,i), fieldtype(T,i))

        if typeof(arg.codegen.typ) == LLVM.PointerType
            # Process pointer to structs as argument buffers
            if typeof(eltype(arg.codegen.typ)) == LLVM.StructType
                argbuf_info = Metadata[]

                # Get information about struct elements first
                new_codegen_typ = eltype(arg.codegen.typ)
                struct_info = add_md((typ=arg.typ, codegen=(typ=new_codegen_typ, i=1)), level+1)

                # Create argument buffer's type metadata
                struct_type_info = Metadata[]

                # Struct element metadata format:
                    # If element is itself a struct directly (not a reference to a struct)
                        # air.struct_type_info keyword
                        # Metadata node of the struct (element struct - NOT a self-reference)
                    # Offset in bytes from start of struct
                    # Size of element in bytes (8 for buffers (pointer size))
                    # Length of element (0 for buffers...always?)
                    # Field type
                    # Field name
                    # Field argument type? (mainly air.indirect_argument)
                        # With structs with the threadgroup addresspace, this metadata node is not present
                        # Because each threadgroup gets the struct data directly without redirection??
                    # If the element is itself a struct directly (not a reference to a struct)
                        # Location index of the element struct in the higher-level struct
                    # Metadata node to more details about element
                    # If argument is unused
                        # air.arg_unused

                # Return the struct element type name and field name from metadata
                function parse_struct_names(md)
                    for (i, item) in enumerate(md)
                        if typeof(item) == MDString && string(item) == "air.arg_type_name"
                            return (md[i+1], md[i+3])
                        end
                    end
                    error("Struct element metadata keyword 'air.arg_type_name' not found in $(md)")
                end

                for (i,struct_field_info) in enumerate(struct_info)

                    type_name, field_name = parse_struct_names(struct_field_info)
                    field_is_struct = typeof(arg.typ) == LLVM.StructType

                    if field_is_struct
                        push!(struct_type_info, MDString("air.struct_type_info"; ctx))
                        push!(struct_type_info, MDNode(struct_field_info; ctx))
                    end

                    push!(struct_type_info, Metadata(ConstantInt(Int32(fieldoffset(arg.typ,i)); ctx)))

                    # If field is (arg)buffer, set size to 8 and length to 0
                    if string(struct_field_info[2]) in ["air.buffer", "air.indirect_buffer"]
                        push!(struct_type_info, Metadata(ConstantInt(Int32(8); ctx))) # Field element size
                        push!(struct_type_info, Metadata(ConstantInt(Int32(0); ctx))) # Length of field
                    else
                        field_type = fieldtype(arg.typ, i)
                        push!(struct_type_info, Metadata(ConstantInt(Int32(sizeof(eltype(field_type))); ctx))) # Field element size
                        push!(struct_type_info, Metadata(ConstantInt(Int32(length(field_type.parameters)); ctx))) # Length of field
                    end
                    push!(struct_type_info, type_name) # Field type
                    push!(struct_type_info, field_name) # Field name
                    push!(struct_type_info, MDString("air.indirect_argument"; ctx))

                    if field_is_struct
                        push!(struct_type_info, Metadata(ConstantInt(Int32(i); ctx)))
                    else
                        struct_field_info = MDNode(struct_field_info; ctx)
                        push!(struct_type_info, struct_field_info)
                    end
                end

                struct_type_info = MDNode(struct_type_info; ctx)

                # Add argument buffer details
                # Create the argument buffer main metadata
                push!(argbuf_info, Metadata(ConstantInt(Int32(arg.codegen.i-1); ctx))) # Argument index
                push!(argbuf_info, MDString("air.indirect_buffer"; ctx))
                push!(argbuf_info, MDString("air.buffer_size"; ctx))
                push!(argbuf_info, Metadata(ConstantInt(Int32(sizeof(arg.typ)); ctx)))
                push!(argbuf_info, MDString("air.location_index"; ctx))
                push!(argbuf_info, Metadata(ConstantInt(Int32(arg.codegen.i-1); ctx)))
                push!(argbuf_info, Metadata(ConstantInt(Int32(1); ctx)))
                # TODO: Check for const array and put to air.read
                push!(argbuf_info, MDString("air.read_write"; ctx))
                push!(argbuf_info, MDString("air.struct_type_info"; ctx))
                push!(argbuf_info, struct_type_info) # Argument buffer type info
                push!(argbuf_info, MDString("air.arg_type_size"; ctx))
                push!(argbuf_info, Metadata(ConstantInt(Int32(sizeof(arg.typ)); ctx))) # Arg type size
                push!(argbuf_info, MDString("air.arg_type_align_size"; ctx))
                push!(argbuf_info, Metadata(ConstantInt(Int32(Base.datatype_alignment(arg.typ)); ctx)))
                push!(argbuf_info, MDString("air.arg_type_name"; ctx))
                push!(argbuf_info, MDString(string(arg.typ); ctx))
                push!(argbuf_info, MDString("air.arg_name"; ctx))
                push!(argbuf_info, MDString("arg_$(arg.codegen.i-1)"; ctx)) # TODO: How to get this? Does the compiler job have it somewhere?
                # Ignore unused flag for now

                # Make argument buffer metadata node
                argbuf_info = MDNode(argbuf_info; ctx)
                return argbuf_info

            # Process simple pointer as simple buffer
            else
                ptr_datatype = arg.typ.parameters[1]

                arg_info_ptr = Metadata[]
                push!(arg_info_ptr, Metadata(ConstantInt(Int32(arg.codegen.i-1); ctx))) # Argument index
                push!(arg_info_ptr, MDString("air.buffer"; ctx))
                push!(arg_info_ptr, MDString("air.location_index"; ctx))
                push!(arg_info_ptr, Metadata(ConstantInt(Int32(arg.codegen.i-1); ctx)))
                push!(arg_info_ptr, Metadata(ConstantInt(Int32(addrspace(arg.codegen.typ)); ctx))) # Address space
                push!(arg_info_ptr, MDString("air.read_write"; ctx)) # TODO: Check for const array
                push!(arg_info_ptr, MDString("air.arg_type_size"; ctx))
                push!(arg_info_ptr, Metadata(ConstantInt(Int32(sizeof(ptr_datatype)); ctx))) # TODO: Get properly
                push!(arg_info_ptr, MDString("air.arg_type_align_size"; ctx))
                push!(arg_info_ptr, Metadata(ConstantInt(Int32(Base.datatype_alignment(ptr_datatype)); ctx)))
                push!(arg_info_ptr, MDString("air.arg_type_name"; ctx))
                # Handle naming for pointer to ArrayType
                if typeof(eltype(arg.codegen.typ)) == LLVM.ArrayType
                    arg_type_name = jl_type_to_c[eltype(eltype(arg.typ))] * string(length(eltype(arg.codegen.typ)))
                else
                    arg_type_name = string(eltype(arg.codegen.typ))
                end
                push!(arg_info_ptr, MDString(arg_type_name; ctx)) # TODO: Get properly
                push!(arg_info_ptr, MDString("air.arg_name"; ctx))
                # TODO: Properly get top-level argument names
                arg_name = field_info != nothing ? string(field_info[2]) : "arg_$(arg.codegen.i)"
                push!(arg_info_ptr, MDString(arg_name; ctx))
                return arg_info_ptr
            end

        elseif typeof(arg.codegen.typ) == LLVM.StructType
            arg_info_struct = []
            for (i, elem) in enumerate(collect(elements(arg.codegen.typ)))
                field_info = structinfo(arg.typ, i)
                push!(arg_info_struct, add_md((typ=field_info[3], codegen=(typ=elem, i=i)), field_info, level+1))
            end
            return arg_info_struct

        # Process as basic leaf argument
        else
            arg_info = Metadata[]

            # If it's a top-level arg, encode as a buffer
            if level == 1

                ## Simple Buffer Argument Metadata layout
                    # Kernel argument index
                    # air.buffer keyword
                    # air.location_index keyword
                    # Kernel argument location index (NOT the same as argument index)
                        # Values are determined as explained by pages 46 and 79 of the Metal docs
                            # https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
                        # Note that these indices are unique to each resource group type [buffer, threadgroup, sampler, texture]
                    # Unknown value - Has always been 1
                        # Vertex/stag_in info? Something with rasters?
                    # Resource usage status
                    # air.arg_type_size keyword
                    # Buffer element size
                    # air.arg_type_align_size keyword
                    # Buffer element alignment
                    # air.arg_type_name keyword
                    # Buffer element type name
                    # air.arg_name keyword
                    # Kernel argument name

                datatype = arg.typ

                arg_info_ptr = Metadata[]
                push!(arg_info_ptr, Metadata(ConstantInt(Int32(arg.codegen.i-1); ctx)))
                push!(arg_info_ptr, MDString("air.buffer"; ctx))
                push!(arg_info_ptr, MDString("air.location_index"; ctx))
                push!(arg_info_ptr, Metadata(ConstantInt(Int32(arg.codegen.i-1); ctx)))
                push!(arg_info_ptr, Metadata(ConstantInt(Int32(1); ctx)))
                push!(arg_info_ptr, MDString("air.read_write"; ctx)) # TODO: Check for const array
                push!(arg_info_ptr, MDString("air.arg_type_size"; ctx))
                push!(arg_info_ptr, Metadata(ConstantInt(Int32(sizeof(datatype)); ctx)))
                push!(arg_info_ptr, MDString("air.arg_type_align_size"; ctx))
                push!(arg_info_ptr, Metadata(ConstantInt(Int32(Base.datatype_alignment(datatype)); ctx)))
                push!(arg_info_ptr, MDString("air.arg_type_name"; ctx))
                push!(arg_info_ptr, MDString(string(eltype(arg.codegen.typ)); ctx))
                push!(arg_info_ptr, MDString("air.arg_name"; ctx))
                # TODO: Properly get top-level argument names
                arg_name = field_info != nothing ? string(field_info[2]) : "arg_$(arg.codegen.i)"
                push!(arg_info_ptr, MDString(arg_name; ctx))

            # Else process as indirect_constant (vector type)
            # TODO: Need to check upstream that we're only passing valid vector types
            elseif typeof(arg.codegen.typ) == LLVM.ArrayType
                # Ensure valid length (<5)
                arg_length = length(arg.codegen.typ)
                arg_length > 4 && error("Invalid Metal kernel ArrayType argument length of $arg_length")

                push!(arg_info, Metadata(ConstantInt(Int32(arg.codegen.i-1); ctx)))
                push!(arg_info, MDString("air.indirect_constant"; ctx))
                push!(arg_info, MDString("air.location_index"; ctx))
                push!(arg_info, Metadata(ConstantInt(Int32(arg.codegen.i-1); ctx)))
                push!(arg_info, Metadata(ConstantInt(Int32(1); ctx)))
                push!(arg_info, MDString("air.arg_type_name"; ctx))
                arg_type_name = jl_type_to_c[eltype(arg.typ)] * string(arg_length)
                push!(arg_info, MDString(arg_type_name; ctx))
                push!(arg_info, MDString("air.arg_name"; ctx))
                arg_name = field_info != nothing ? string(field_info[2]) : "arg_$(arg.codegen.i)"
                push!(arg_info, MDString(arg_name; ctx))
            else
                error("Unknown Metal kernel argument type $(arg.typ)")
            end

            return arg_info
        end
    end

    ## argument info
    arg_infos = Metadata[]

    # Iterate through arguments and create metadata for them
    for arg in classify_arguments(job, eltype(llvmtype(entry)))
        # Ignore ghost type
        if arg.cc != GHOST
            arg_info = add_md(arg)
            # Ensure returned type is a metadata node
            if typeof(arg_info) != MDTuple
                arg_info = MDNode(arg_info; ctx)
            end
            push!(arg_infos, arg_info)
        end
    end

    # Create metadata for argument intrinsics last
    for (i, intr_fn) in enumerate(used_intrinsics)
        arg_info = Metadata[]
        push!(arg_info, Metadata(ConstantInt(Int32(length(parameters(entry))-i); ctx)))
        push!(arg_info, MDString("air." * kernel_intrinsics[intr_fn].air_name; ctx))
        push!(arg_info, MDString("air.arg_type_name"; ctx))
        push!(arg_info, MDString(kernel_intrinsics[intr_fn].air_typ; ctx))
        push!(arg_info, MDString("air.arg_name"; ctx))
        push!(arg_info, MDString(kernel_intrinsics[intr_fn].air_name; ctx))
        arg_info = MDNode(arg_info; ctx)
        push!(arg_infos, arg_info)
    end
    arg_infos = MDNode(arg_infos; ctx)
    ## stage info
    stage_infos = Metadata[]
    stage_infos = MDNode(stage_infos; ctx)

    kernel_md = MDNode([entry, stage_infos, arg_infos]; ctx)
    push!(metadata(mod)["air.kernel"], kernel_md)

    return
end

@unlocked function mcgen(job::CompilerJob{MetalCompilerTarget}, mod::LLVM.Module,
                         format=LLVM.API.LLVMObjectFile)
    # translate to SPIR-V
    input = tempname(cleanup=false) * ".bc"
    translated = tempname(cleanup=false) * ".metallib"
    write(input, mod)
    Metal_LLVM_Tools_jll.metallib_as() do assembler
        proc = run(ignorestatus(`$assembler -o $translated $input`))
        if !success(proc)
            error("""Failed to translate LLVM code to MetalLib.
                     If you think this is a bug, please file an issue and attach $(input).""")
        end
    end

    output = if format == LLVM.API.LLVMObjectFile
        read(translated)
    else
        # disassemble
        Metal_LLVM_Tools_jll.metallib_dis() do disassembler
            read(`$disassembler -o - $translated`, String)
        end
    end

    rm(input)
    rm(translated)

    return output
end
