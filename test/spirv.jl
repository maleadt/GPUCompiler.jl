@testset "SPIR-V" begin

include("definitions/spirv.jl")

############################################################################################

@testset "IR" begin

@testset "kernel functions" begin
@testset "calling convention" begin
    kernel() = return

    ir = sprint(io->spirv_code_llvm(io, kernel, Tuple{}; dump_module=true))
    @test !occursin("spir_kernel", ir)

    ir = sprint(io->spirv_code_llvm(io, kernel, Tuple{};
                                    dump_module=true, kernel=true))
    @test occursin("spir_kernel", ir)
end

@testset "byval workaround" begin
    kernel(x) = return

    ir = sprint(io->spirv_code_llvm(io, kernel, Tuple{Tuple{Int}}))
    @test occursin(r"@.*julia_.+_kernel.+\(({ i64 }|\[1 x i64\])\*", ir)

    ir = sprint(io->spirv_code_llvm(io, kernel, Tuple{Tuple{Int}}; kernel=true))
    @test occursin(r"@.*julia_.+_kernel.+\({ ({ i64 }|\[1 x i64\]) }\*.+byval", ir)
end
end

end

############################################################################################

@testset "asm" begin

@testset "trap removal" begin
    function kernel(x)
        x && error()
        return
    end

    spirv_code_native(devnull, kernel, Tuple{Bool}; kernel=true)
    # TODO: test for error reporting once we've implemented that
end

end

############################################################################################

end
