#pragma once
#include "alice/alice.hpp"
#include "include/database/network/aig_network.hpp"
#include "include/database/network/klut_network.hpp"
#include "include/database/views/depth_view.hpp"
#include <array>

namespace alice 
{

class print_stats_command : public command 
{
public:
    explicit print_stats_command(const environment::ptr& env) : command(env, "print the stats for current Network!") 
    {
        add_option("--type, -t", type, "0 for AIG, 1 for FPGA, 2 for ASIC, [default = 0]");
    }

    rules validity_rules() const { return {}; }
protected:
    void execute()
    {
        if(!is_set("-t")) {
            type = 0;
        }
        if( type == 0 ) {
            if( store<iFPGA::aig_network>().empty() ) {
                printf("WARN: there is no any stored AIG file, please refer to the command \"read_aiger\"\n");
                return;
            }
            iFPGA::aig_network aig = store<iFPGA::aig_network>().current()._storage;
            iFPGA::depth_view<iFPGA::aig_network> daig(aig);
            uint64_t inv_edges = 0u;
            uint64_t po_inv = 0u;
            uint64_t fanout_sum = 0u;
            uint32_t fanout_max = 0u;
            uint32_t fanout_ge4 = 0u;
            uint64_t level_sum = 0u;
            uint32_t level_ge_half = 0u;
            const auto depth = daig.depth();

            aig.foreach_gate( [&]( auto const& n ) {
                const auto fanout = aig.fanout_size( n );
                fanout_sum += fanout;
                fanout_max = std::max( fanout_max, fanout );
                if ( fanout >= 4u ) {
                    ++fanout_ge4;
                }

                const auto level = daig.level( n );
                level_sum += level;
                if ( level * 2u >= std::max( 1u, depth ) ) {
                    ++level_ge_half;
                }

                aig.foreach_fanin( n, [&]( auto const& f ) {
                    if ( aig.is_complemented( f ) ) {
                        ++inv_edges;
                    }
                } );
            } );

            aig.foreach_po( [&]( auto const& f ) {
                if ( aig.is_complemented( f ) ) {
                    ++po_inv;
                }
            } );

            printf(
                "Stats of AIG: pis=%d, pos=%d, area=%d, depth=%d, inv=%llu, po_inv=%llu, fanout_sum=%llu, fanout_max=%u, fanout_ge4=%u, level_sum=%llu, level_ge_half=%u\n",
                aig.num_pis(),
                aig.num_pos(),
                aig.num_gates(),
                depth,
                static_cast<unsigned long long>( inv_edges ),
                static_cast<unsigned long long>( po_inv ),
                static_cast<unsigned long long>( fanout_sum ),
                fanout_max,
                fanout_ge4,
                static_cast<unsigned long long>( level_sum ),
                level_ge_half
            );
        }
        else if (type == 1) {
            if( store<iFPGA::klut_network>().empty() ) {
                printf("WARN: there is no any FPGA mapping result, please refer to the command \"map_fpga\"\n");
                return;
            }
            iFPGA::klut_network klut = store<iFPGA::klut_network>().current()._storage;
            iFPGA::depth_view<iFPGA::klut_network> dklut(klut);
            std::array<uint32_t, 7> lut_hist{};
            uint64_t fanout_sum = 0u;
            uint32_t fanout_max = 0u;
            uint32_t fanout_ge4 = 0u;
            uint64_t level_sum = 0u;
            uint32_t level_ge_half = 0u;
            const auto depth = dklut.depth();

            klut.foreach_gate( [&]( auto const& n ) {
                const auto fanin = std::min<uint32_t>( 6u, klut.fanin_size( n ) );
                ++lut_hist[fanin];

                const auto fanout = klut.fanout_size( n );
                fanout_sum += fanout;
                fanout_max = std::max( fanout_max, fanout );
                if ( fanout >= 4u ) {
                    ++fanout_ge4;
                }

                const auto level = dklut.level( n );
                level_sum += level;
                if ( level * 2u >= std::max( 1u, depth ) ) {
                    ++level_ge_half;
                }
            } );

            printf(
                "Stats of FPGA: pis=%d, pos=%d, area=%d, depth=%d, lut1=%u, lut2=%u, lut3=%u, lut4=%u, lut5=%u, lut6=%u, fanout_sum=%llu, fanout_max=%u, fanout_ge4=%u, level_sum=%llu, level_ge_half=%u\n",
                klut.num_pis(),
                klut.num_pos(),
                klut.num_gates(),
                depth,
                lut_hist[1],
                lut_hist[2],
                lut_hist[3],
                lut_hist[4],
                lut_hist[5],
                lut_hist[6],
                static_cast<unsigned long long>( fanout_sum ),
                fanout_max,
                fanout_ge4,
                static_cast<unsigned long long>( level_sum ),
                level_ge_half
            );
        }
        else if(type == 2) {
            printf("FAIL, comming soon!\n");
            return;
        }
        else {
            printf("FAIL, not supported type!\n");
            return;
        }
    }
private:
    int type = 0;
};
ALICE_ADD_COMMAND(print_stats, "Utils");
};
