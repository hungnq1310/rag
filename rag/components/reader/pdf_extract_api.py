import logging
import os.path
from pathlib import Path
from typing import (
    Union
)
import dotenv
import glob

from adobe.pdfservices.operation.auth.credentials import Credentials
from adobe.pdfservices.operation.exception.exceptions import ServiceApiException, ServiceUsageException, SdkException
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_pdf_options import ExtractPDFOptions
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_element_type import ExtractElementType
from adobe.pdfservices.operation.pdfops.options.extractpdf.extract_renditions_element_type import ExtractRenditionsElementType
from adobe.pdfservices.operation.pdfops.options.extractpdf.table_structure_type import TableStructureType
from adobe.pdfservices.operation.execution_context import ExecutionContext
from adobe.pdfservices.operation.io.file_ref import FileRef
from adobe.pdfservices.operation.pdfops.extract_pdf_operation import ExtractPDFOperation

# Get the samples from http://www.adobe.com/go/pdftoolsapi_python_sample
# Run the sample:
# python src/extractpdf/extract_txt_table_info_with_char_bounds_from_pdf.py

# Load credentials and loglevel
dotenv.load_dotenv()
PDF_SERVICES_CLIENT_ID=os.getenv("PDF_SERVICES_CLIENT_ID")
PDF_SERVICES_CLIENT_SECRET=os.getenv("PDF_SERVICES_CLIENT_SECRET")
LOGLEVEL=os.getenv("LOGLEVEL", "INFO")


logging.getLogger().setLevel(LOGLEVEL)
logger = logging.getLogger(__name__)


class PDFExtractAPI:
    def __init__(self, 
                 config,
                 params,) -> None:
        self.config = config
        self.params = params
        self.credential = self._build_credentials()
        self.extract_pdf_options = self._build_extract_options()


    def _build_credentials(self) -> Credentials:
        # Validate environmental variables
        if PDF_SERVICES_CLIENT_ID is not None or PDF_SERVICES_CLIENT_SECRET is not None:
            logging.error("Extract PDF Services can not create due to not having `PDF_SERVICES_CLIENT_ID` or `PDF_SERVICES_CLIENT_SECRET`")
            raise ServiceApiException
        # Initial setup, create credentials instance.
        credentials = Credentials.service_principal_credentials_builder(). \
            with_client_id(PDF_SERVICES_CLIENT_ID). \
            with_client_secret(PDF_SERVICES_CLIENT_SECRET). \
            build()
        return credentials
    

    def _operate(self) -> None:
        """Create an ExecutionContext using credentials and create a new operation instance.
        """
        try:
            #get base path.
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

            # get pdf files
            files = glob.glob(base_path + "/data/*.pdf")

            #Create an ExecutionContext using credentials .
            execution_context = ExecutionContext.create(self.credential)

            # Start to extract
            for idx, file in enumerate(files):
                # create a new operation instance
                extract_pdf_operation = ExtractPDFOperation.create_new()

                #Set operation input from a source file.
                source = FileRef.create_from_local_file(file)
                extract_pdf_operation.set_input(source)

                # set extract pdf options
                extract_pdf_operation.set_options(self.extract_pdf_options)

                #Execute the operation.
                result: FileRef = extract_pdf_operation.execute(execution_context)
                logger.debug(result)

                #Save the result to the specified location.
                result.save_as(base_path + f"/output/ExtractFromPDF_{idx}.zip")
                logger.info("Save success!")
        except (ServiceApiException, ServiceUsageException, SdkException):
            logging.exception("Exception encountered while executing operation")


    def _build_extract_options(self) -> ExtractPDFOptions:
        # Build ExtractPDF options and set them into the operation
        extract_pdf_options: ExtractPDFOptions = ExtractPDFOptions.builder()
        # config options

        if self.params.only_extract_text: # element_to_extract list
            extract_pdf_options.with_element_to_extract(ExtractElementType.TEXT)
        else:
            extract_pdf_options.with_elements_to_extract([ExtractElementType.TEXT, ExtractElementType.TABLES])
        if self.params.only_render_image:
            extract_pdf_options.with_elements_to_extract_renditions(ExtractRenditionsElementType.FIGURES)
        else:
            extract_pdf_options.with_elements_to_extract_renditions(
                [ExtractRenditionsElementType.TABLES,
                 ExtractRenditionsElementType.FIGURES]
            )
        if self.params.with_get_char_info:
            extract_pdf_options.with_get_char_info(True)
        if self.params.with_table_structure:
            extract_pdf_options.with_table_structure_format(TableStructureType.CSV)
        if self.params.include_styling_info:
            extract_pdf_options.with_include_styling_info(True)
        # Build
        extract_pdf_options = extract_pdf_options.build()
        return extract_pdf_options

    