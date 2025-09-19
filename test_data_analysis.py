import unittest
from pathlib import Path
import shutil
import json
from data_analysis import DataAnalyzer

class TestDataAnalyzer(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        """Set up a temporary dataset for testing"""
        cls.test_dataset_path = Path("test_dataset")
        cls.test_output_path = Path("test_output")
        
        # Create test dataset structure
        cls.test_dataset_path.mkdir(exist_ok=True)
        categories = ["Mandatory_Traffic_Signs", "Cautionary_Traffic_Signs", "Informatory_Traffic_Signs"]
        for category in categories:
            category_path = cls.test_dataset_path / category
            category_path.mkdir(exist_ok=True)
            for i in range(3):  # Create 3 classes per category
                class_path = category_path / f"class_{i}"
                class_path.mkdir(exist_ok=True)
                for j in range(5):  # Add 5 images per class
                    (class_path / f"image_{j}.jpg").touch()

    @classmethod
    def tearDownClass(cls):
        """Clean up the temporary dataset and output directory"""
        shutil.rmtree(cls.test_dataset_path)
        shutil.rmtree(cls.test_output_path)

    def test_analyze_dataset(self):
        """Test the analyze_dataset method"""
        analyzer = DataAnalyzer(dataset_path=self.test_dataset_path, output_path=self.test_output_path)
        summary = analyzer.analyze_dataset()
        
        # Check summary structure
        self.assertIn('total_images', summary)
        self.assertIn('total_classes', summary)
        self.assertIn('statistics', summary)
        self.assertEqual(summary['total_images'], 45)  # 3 categories * 3 classes * 5 images
        self.assertEqual(summary['total_classes'], 9)  # 3 categories * 3 classes

    def test_generate_summary_report(self):
        """Test the _generate_summary_report method"""
        analyzer = DataAnalyzer(dataset_path=self.test_dataset_path, output_path=self.test_output_path)
        analyzer.analyze_dataset()
        
        summary_path = analyzer.analysis_dir / 'dataset_summary.json'
        self.assertTrue(summary_path.exists())
        
        with open(summary_path, 'r') as f:
            summary = json.load(f)
        
        self.assertIn('total_images', summary)
        self.assertIn('statistics', summary)
        self.assertIn('imbalanced_classes', summary)

    def test_get_augmentation_recommendations(self):
        """Test the get_augmentation_recommendations method"""
        analyzer = DataAnalyzer(dataset_path=self.test_dataset_path, output_path=self.test_output_path)
        recommendations = analyzer.get_augmentation_recommendations()
        
        self.assertIn('oversampling_needed', recommendations)
        self.assertIn('undersampling_needed', recommendations)
        self.assertIsInstance(recommendations['oversampling_needed'], list)
        self.assertIsInstance(recommendations['undersampling_needed'], list)

if __name__ == "__main__":
    unittest.main()
