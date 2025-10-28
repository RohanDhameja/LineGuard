"""
Generate Synthetic Datasets for Fire App
Run this script to create realistic LIDAR and vegetation spread data
"""

from utils.synthetic_data_generator import generate_all_datasets, SyntheticDataGenerator


# Define zones (same as in app.py)
ZONES = [
    {"id": 0, "min_lat": 35.7674275193022, "max_lat": 35.7674275193022, "min_lon": -120.442012631, "max_lon": -120.442012631},
    {"id": 1, "min_lat": 35.7868696904087, "max_lat": 35.7868696904087, "min_lon": -120.418170521401, "max_lon": -120.418170521401},
    {"id": 2, "min_lat": 35.8489547981213, "max_lat": 35.8489547981213, "min_lon": -120.348411490484, "max_lon": -120.348411490484},
    {"id": 3, "min_lat": 34.2239140945453, "max_lat": 34.2239140945453, "min_lon": -116.90996414966, "max_lon": -116.90996414966},
    {"id": 4, "min_lat": 34.2106297635416, "max_lat": 34.2106297635416, "min_lon": -116.906609995797, "max_lon": -116.906609995797},
    {"id": 5, "min_lat": 34.1976199560805, "max_lat": 34.1976199560805, "min_lon": -116.909203216427, "max_lon": -116.909203216427},
    {"id": 6, "min_lat": 36.6457420953003, "max_lat": 36.6457420953003, "min_lon": -120.972005184682, "max_lon": -120.972005184682},
    {"id": 7, "min_lat": 36.6432756517884, "max_lat": 36.6432756517884, "min_lon": -120.947377254923, "max_lon": -120.947377254923},
]


def main():
    """Generate all synthetic datasets"""
    print("\n" + "="*70)
    print("🌲 SYNTHETIC DATASET GENERATOR - FIRE APP")
    print("="*70)
    
    # Generate datasets
    lidar_data, spread_data, growth_simulation = generate_all_datasets(
        zones=ZONES,
        weeks=8,
        random_seed=42
    )
    
    # Save to files
    print("\n💾 Saving datasets...")
    generator = SyntheticDataGenerator()
    generator.save_datasets(lidar_data, spread_data, growth_simulation, output_dir='data')
    
    # Print summary
    print("\n" + "="*70)
    print("📊 DATASET SUMMARY")
    print("="*70)
    
    print("\n🌲 LIDAR Canopy Height Data:")
    print(f"   Zones: {len(lidar_data)}")
    print(f"   Sample points per zone: 100 (10x10 grid)")
    
    for zone_id, data in lidar_data.items():
        stats = data['statistics']
        print(f"\n   Zone {zone_id} ({data['vegetation_type']}):")
        print(f"      Mean height: {stats['mean_height']:.2f}m")
        print(f"      Max height: {stats['max_height']:.2f}m")
        print(f"      Coverage: {data['coverage_percentage']:.1f}%")
    
    print("\n🌿 Vegetation Spread Data:")
    print(f"   Zones: {len(spread_data)}")
    print(f"   Weeks: 8")
    
    for zone_id, data in spread_data.items():
        print(f"\n   Zone {zone_id}:")
        print(f"      Initial coverage: {data['initial_coverage']:.1f}%")
        print(f"      Spread rate: {data['spread_rate']:.2f}%/week")
        print(f"      Pattern: {data['spread_pattern']}")
    
    print("\n📈 Growth Simulation Data:")
    print(f"   Zones: {len(growth_simulation)}")
    print(f"   Weeks: 8")
    
    for zone_id, data in growth_simulation.items():
        final_height = data['weekly_heights'][-1]['cumulative_height']
        total_growth = final_height - data['initial_height']
        print(f"\n   Zone {zone_id} ({data['vegetation_type']}):")
        print(f"      Initial height: {data['initial_height']:.2f}m")
        print(f"      Final height (Week 7): {final_height:.2f}m")
        print(f"      Total growth: {total_growth:.2f}m")
        print(f"      Growth rate: {data['growth_parameters']['growth_rate_type']}")
    
    print("\n" + "="*70)
    print("✅ DATASETS GENERATED SUCCESSFULLY!")
    print("="*70)
    
    print("\n📁 Files created:")
    print("   - data/lidar_canopy_data.json")
    print("   - data/vegetation_spread_data.json")
    print("   - data/growth_simulation_data.json")
    
    print("\n🚀 Next steps:")
    print("   1. Check the 'data/' directory for generated files")
    print("   2. Run the app: python3 app.py")
    print("   3. The app will automatically use this realistic data!")
    
    print("\n💡 To regenerate with different random data:")
    print("   python3 generate_datasets.py")
    print()


if __name__ == "__main__":
    main()

